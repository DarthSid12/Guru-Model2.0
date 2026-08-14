class LogPolar(torch.nn.Module):
    def __init__(self, input_shape=None, output_shape=None, smoothing = 0, 
                 mask = False, position='circumscribed', log_polar_distance = 2, random_center = False,
                 device = 'cpu', mapping="log", power_gamma=0.17, power_offset=0.01):
        super().__init__()
        self.device = device
        self.input_shape = input_shape
        self.default_center = input_shape[0] / 2, input_shape[1] / 2

        self.output_shape = output_shape
        self.smoothing = smoothing
        self.mask = mask
        self.position = position

        self.log_polar_distance = log_polar_distance
        self.random_center = random_center

        self.mapping = mapping
        self.power_gamma = power_gamma
        self.power_offset = power_offset

        X, Y = self.compute_map(self.input_shape, self.output_shape)
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)        

    def getPoints(self, numPoints, prob_arr, threshold = 0.20):
        crop_size = 0
        prob_reshape = prob_arr.reshape(-1)
        
        y_threshold_amt = max(crop_size // 2, int(threshold * prob_arr.shape[0]))
        x_threshold_amt = max(crop_size // 2, int(threshold * prob_arr.shape[0]))
        border_mask = np.zeros_like(prob_arr)
        border_mask[y_threshold_amt:-y_threshold_amt, x_threshold_amt:-x_threshold_amt] = 1
        border_mask = border_mask.reshape(-1)

        prob_border_masked = prob_reshape * border_mask
        prob_border_masked /= prob_border_masked.sum()

        try:
            points = np.random.choice(prob_reshape.shape[0], numPoints, p = prob_border_masked)
            unraveled_points = np.array(np.unravel_index(points, prob_arr.shape))
            return unraveled_points
        except:
            print("errors")
            return np.random.choice(prob_reshape.shape[0], numPoints)
                
        
    def SaliencePoints(self, data):
        
        cv2_img = cv2.cvtColor(np.array(data).transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(cv2_img)
        points = self.getPoints(1, saliencyMap)
        return (points[0][0], points[1][0])
    
    def compute_map(self, input_shape, output_shape):
        if self.position == 'circumscribed':
            max_radius = (
                torch.tensor(
                    input_shape,
                    device=self.device
                ).float().norm() / 2
            ) * self.log_polar_distance
        else:
            max_radius = (
                torch.tensor(
                    input_shape,
                    device=self.device
                ).float().max() / 2
            ) * self.log_polar_distance

        theta, r = torch.meshgrid(
            torch.arange(
                self.output_shape[0],
                device=self.device
            ),
            torch.arange(
                self.output_shape[1],
                device=self.device
            ),
            indexing='ij'
        )

        theta = theta.float()
        r = r.float()

        # Normalize radius to [0,1]
        E = r / (self.output_shape[1] - 1)

        # -----------------------------------
        # LOG POLAR
        # -----------------------------------

        if self.mapping == "log":
            MAX_R = torch.log(max_radius)

            rho = torch.exp(
                E * MAX_R
            )

        # -----------------------------------
        # POWER LAW
        # -----------------------------------

        elif self.mapping == "power":
            gamma = self.power_gamma
            a = self.power_offset

            rho = (E + a) ** gamma

            # normalize to image radius

            rho_min = a ** gamma
            rho_max = (1 + a) ** gamma

            rho = (
                (rho - rho_min)
                / (rho_max - rho_min)
            )

            rho = rho * max_radius

        else:
            raise ValueError(
                f"Unknown mapping {self.mapping}"
            )

        # Convert to x,y
        X = rho * torch.cos(
            theta * 2 * torch.pi
            / self.output_shape[0]
        )

        Y = rho * torch.sin(
            theta * 2 * torch.pi
            / self.output_shape[0]
        )

        return X, Y

    def compute_mask(self, X, Y, input_shape):
        return (0 <= X) & (X < input_shape[0]) & (0 <= Y) & (Y < input_shape[1])
    
    def forward(self, data, center_x = None, center_y = None):
        
        img = data.permute(0, 2, 3, 1)
        img = (img - img.min()) / (img.max() - img.min())
        
        if data.shape[-2:] != self.input_shape:
            X, Y = self.compute_map(data.shape[-2:], self.output_shape)
        else:
            X = self.get_buffer('X')
            Y = self.get_buffer('Y')

        if not center_x or not center_y:
            center_y, center_x = self.default_center
            
        if self.random_center and random.random() > 0.4 :
            center_y, center_x = self.SaliencePoints(data)
            
        X = center_x + X
        Y = center_y - Y

        # print("centre", center_x, center_y )
        mask = (self.compute_mask(X, Y, self.input_shape)  if self.mask else torch.ones_like(X)).to(self.device)
        # print("mask", mask)
        if self.smoothing == None:
            return (
                mask * (
                    data[
                      ...,
                      Y.long().clamp(0, data.shape[-2] - 1),
                      X.long().clamp(0, data.shape[-1] - 1),
                      # Y.long() % (data.shape[-2] - 1),
                      # X.long() % (data.shape[-1] - 1)
                    ]
                )
            )
                
        y_down, x_down = Y.long().clamp(0, data.shape[-2] - 1), X.long().clamp(0, data.shape[-1] - 1)
        y_up, x_up = (y_down+1).clamp(0, data.shape[-2] - 1), (x_down+1).clamp(0, data.shape[-1] - 1)
        
        down_down_dist = (Y - y_down)**self.smoothing + (X - x_down)**self.smoothing
        down_up_dist = (Y - y_down)**self.smoothing + (X - x_up)**self.smoothing
        up_down_dist = (Y - y_up)**self.smoothing + (X - x_down)**self.smoothing
        up_up_dist = (Y - y_up)**self.smoothing + (X - x_up)**self.smoothing

        total_dist = down_down_dist + down_up_dist +  up_down_dist +  up_up_dist
        
        down_down_weight = (down_down_dist / total_dist).to(self.device)
        down_up_weight = (down_up_dist / total_dist).to(self.device)
        up_down_weight = (up_down_dist / total_dist).to(self.device)
        up_up_weight = (up_up_dist / total_dist).to(self.device)

        return (
            mask * (
                down_down_weight * data[...,y_down,x_down] +
                down_up_weight * data[...,y_down,x_up] +
                up_down_weight * data[...,y_up,x_down] +
                up_up_weight * data[...,y_up,x_up]
            )
        )
    
    def forwardReturnMapping(self, data, center_x = None, center_y = None):
        
        img = data.permute(0, 2, 3, 1)
        img = (img - img.min()) / (img.max() - img.min())
        
        if data.shape[-2:] != self.input_shape:
            X, Y = self.compute_map(data.shape[-2:], self.output_shape)
        else:
            X = self.get_buffer('X')
            Y = self.get_buffer('Y')

        if not center_x or not center_y:
            center_y, center_x = self.default_center
            
        if self.random_center and random.random() > 0.4 :
            center_y, center_x = self.SaliencePoints(data)
            
        X = center_x + X
        Y = center_y - Y

        # print("centre", center_x, center_y )
        mask = (self.compute_mask(X, Y, self.input_shape)  if self.mask else torch.ones_like(X)).to(self.device)
        # print("mask", mask)
        if self.smoothing == None:
            return (
                mask * (
                    data[
                      ...,
                      Y.long().clamp(0, data.shape[-2] - 1),
                      X.long().clamp(0, data.shape[-1] - 1),
                      # Y.long() % (data.shape[-2] - 1),
                      # X.long() % (data.shape[-1] - 1)
                    ]
                )
            )

        y_down, x_down = Y.long().clamp(0, data.shape[-2] - 1), X.long().clamp(0, data.shape[-1] - 1)
        y_up, x_up = (y_down+1).clamp(0, data.shape[-2] - 1), (x_down+1).clamp(0, data.shape[-1] - 1)
        
        down_down_dist = (Y - y_down)**self.smoothing + (X - x_down)**self.smoothing
        down_up_dist = (Y - y_down)**self.smoothing + (X - x_up)**self.smoothing
        up_down_dist = (Y - y_up)**self.smoothing + (X - x_down)**self.smoothing
        up_up_dist = (Y - y_up)**self.smoothing + (X - x_up)**self.smoothing

        total_dist = down_down_dist + down_up_dist +  up_down_dist +  up_up_dist
        
        down_down_weight = (down_down_dist / total_dist).to(self.device)
        down_up_weight = (down_up_dist / total_dist).to(self.device)
        up_down_weight = (up_down_dist / total_dist).to(self.device)
        up_up_weight = (up_up_dist / total_dist).to(self.device)

        return (
            mask * (
                down_down_weight * data[...,y_down,x_down] +
                down_up_weight * data[...,y_down,x_up] +
                up_down_weight * data[...,y_up,x_down] +
                up_up_weight * data[...,y_up,x_up]
            ), x_down, y_down
        )