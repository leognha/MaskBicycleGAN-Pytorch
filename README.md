# MaskBicycleGAN-Pytorch
參考BicycleGAN的架構，在訓練BicycleGAN的過程中加入Mask以及在loss function的部分加入了mask loss和content loss，使得生成圖片的品質能夠更好。

此方法用於Sky Finder dataset(https://mypages.valdosta.edu/rpmihail/skyfinder/) 上，解決原本由BicycleGAN所生成的圖片在細節會有模糊以及不夠真實的問題並且增強邊界的穩定性。

可利用此方法作為Data Augmentation來增強Segmentation model的效能。

# 生成圖像的漸進變化
有關雲的多寡:
![image](https://github.com/leognha/MaskBicycleGAN-Pytorch/blob/master/img/D1.png)

有關色調的變化:
![image](https://github.com/leognha/MaskBicycleGAN-Pytorch/blob/master/img/D6.png)

有關亮暗程度的變化:
![image](https://github.com/leognha/MaskBicycleGAN-Pytorch/blob/master/img/D8.png)


# 有關細節的比較
![image](https://github.com/leognha/MaskBicycleGAN-Pytorch/blob/master/img/compare_detail2.png)

![image](https://github.com/leognha/MaskBicycleGAN-Pytorch/blob/master/img/compare_detail2_detail.png)

