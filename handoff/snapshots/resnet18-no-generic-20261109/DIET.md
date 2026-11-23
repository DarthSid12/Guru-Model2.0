# New training diet: no generic houses

All jobs are held so this diet can be reviewed before training starts. Four fresh ResNet18 models: LP/CNN seed45 on Anvil, LP/CNN seed46 on Delta.

| Stage | Epochs | Face identities | Object categories | House identities | Generic-house images |
|---|---|---:|---:|---:|---:|
| 1 | 1–2 | 4 | 0 | 0 | 0 |
| 2 | 3–4 | 8 | 0 | 0 | 0 |
| 3 | 5–8 | 16 | 4 | 0 | 0 |
| 4 | 9–12 | 32 | 8 | 4 | 0 |
| 5 | 13–20 | 64 | 16 | 8 | 0 |
| 6 | 21–28 | 128 | 32 | 16 | 0 |
| 7 | 29–44 | 256 | 64 | 32 | 0 |
| 8 | 45–60 | 512 | 64 | 64 | 0 |
| 9 | 61–92 | 1024 | 64 | 128 | 0 |
| 10 | 93–124 | 2048 | 64 | 128 | 0 |

Counts are cumulative. Source: VGGFace2 faces, the same revised64-object Open Images/Wordbank bank, and the same128 Houses535 identities.

| Category | Identities/categories | Training photos each | Validation photos each | Inverted test photos each |
|---|---:|---:|---:|---:|
| Faces | 2048 | 200 | 12 | 12 |
| Objects | 64 | 500 | 100 | 100 |
| Houses | 128 | 3 | 1 | 1 |

House training views: original,left15°,right15°. Validation:left30°. Inverted classification test:right30°,rotated180°. The five later random synthetic views remain excluded.

Final classifier:2240 classes. Same fixed identity orders. Same Wordbank understands curriculum and approved toy exceptions; no live-animal object categories.

Unchanged: scratch ResNet18,256-unit sigmoid code,T2,dropout.3; AdamW lr.001,weight decay.05,cosine schedule,500-step warmup after each later curriculum stage; batch256,2000steps/epoch,16fixations,32stored coordinates,180px crops,bfloat16 AMP,channels_last,8loader workers,sqrt category sampling,patience20. Generic category is absent from sampling and classifier labels; remaining sampling weights renormalize.

## Exact object introductions

- Training stage3; Wordbank checkpoint8months; total4: ball, bottle, book, car.
- Training stage4; Wordbank checkpoint10months; total8: banana, cup, bed, telephone.
- Training stage5; Wordbank checkpoint12months; total16: shoe, bathtub, sock, balloon, hat, blanket, door, spoon.
- Training stage6; Wordbank checkpoint15months; total32: cookie, apple, tv, chair, bread, teddybear, pants, shirt, stairs, doll, truck, airplane, refrigerator, glasses, train, couch.
- Training stage7; Wordbank checkpoint18months; total64: fork, coat, bowl, table, bus, bicycle, boots, egg, window, pillow, box, carrots, pen, orange_food, plate, pizza, noodles, drawer, ice_cream, sink, motorcycle, glass, clock, sweater, purse, watch_object, cake, dress_object, necklace, candy, lamp, shorts.

Wordbank month labels are curriculum selection checkpoints, not claimed visual-learning ages.

Exact fixed identity order files: [faces](/Users/darthsid/Documents/ChatGPT/GURU/operations/resnet18-no-generic-20261109/faces_vgg2k_order.txt) · [houses](/Users/darthsid/Documents/ChatGPT/GURU/operations/resnet18-no-generic-20261109/house_identity_order.txt).
