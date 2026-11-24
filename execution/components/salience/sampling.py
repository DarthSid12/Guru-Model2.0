"""Independent fixation sampler; coordinates always use Cartesian (x, y)."""
def validate_sampling(c):
    if c['method'] not in ['gabor','uniform','grid']:raise ValueError('method must be gabor, uniform or grid')
    if c['num_points']<1:raise ValueError('num_points must be positive')
    if not 0<=c['edge_margin']<.5:raise ValueError('edge_margin must be in [0, 0.5)')
    if c['batch_size']<1 or c['seed']<0:raise ValueError('batch_size must be positive and seed nonnegative')
    center=c['attention_center']
    if center is not None and (len(center)!=2 or not all(0<=x<=1 for x in center)):raise ValueError('attention_center is null or normalized [x,y] in [0,1]')
    if center is not None and c['method']!='gabor':raise ValueError('attention_center is a Gabor attention-prior control only')

def sample(images,c,pipeline=None):
    import math
    import torch
    n,_,h,w=images.shape;k=c['num_points'];margin=c['edge_margin']
    if c['method']=='gabor':
        if pipeline is None:
            from training.salience_trans import SaliencePipeline
            pipeline=SaliencePipeline(type='valid',device=c['device'],num_salient_points=k,edge_margin=margin).to(c['device'])
        center=c['attention_center'];center=None if center is None else (center[0]*(w-1),center[1]*(h-1))
        return pipeline.sample_salience_points(images,center=center)
    xs=torch.arange(math.ceil(margin*w),min(w,math.floor((1-margin)*w)+1),device=images.device)
    ys=torch.arange(math.ceil(margin*h),min(h,math.floor((1-margin)*h)+1),device=images.device)
    yy,xx=torch.meshgrid(ys,xs,indexing='ij');points=torch.stack([xx.flatten(),yy.flatten()],1)
    if k>len(points):raise ValueError('Requested more points than pixels in the allowed region')
    if c['method']=='uniform':return torch.stack([points[torch.randperm(len(points),device=images.device)[:k]] for _ in range(n)])
    # A near-square Cartesian lattice within the requested region.
    cols=math.ceil(math.sqrt(k*w/h));rows=math.ceil(k/cols)
    gx=torch.linspace(float(xs[0]),float(xs[-1]),cols,device=images.device).round().long()
    gy=torch.linspace(float(ys[0]),float(ys[-1]),rows,device=images.device).round().long()
    yy,xx=torch.meshgrid(gy,gx,indexing='ij');grid=torch.stack([xx.flatten(),yy.flatten()],1)[:k]
    if len(torch.unique(grid,dim=0))!=k:raise ValueError('Grid is too dense for this image region')
    return grid.unsqueeze(0).repeat(n,1,1)
