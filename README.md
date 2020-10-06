# Pytorch-BPR

Note that I use the two sub datasets provided by Xiangnan's [repo](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data). Another pytorch NCF implementaion can be found at this [repo](https://github.com/guoyang9/NCF).

I utilized a factor number **32**, and posted the results in the NCF paper and this implementation here. Since there is no specific numbers in their paper, I found this implementation achieved a better performance than the original curve. Moreover, the batch_size is not very sensitive with the final model performance.

Models 			| MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10
------ 			| --------------- | ----------------- | --------------- | -----------------
pytorch-BPR    	| 0.700 		  | 0.418             | 0.877 			| 0.551

## Install

```bash
make install
```

The key requirements installed through conda include:

```yaml
- python>=3.8
- pandas>=1.1
- numpy>=1.19
- pytorch>=1.6
- tensorboardX>=2.1 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)
```

## Uninstall

```bash
make uninstall
```

## Example to run

```bash
python main.py --factor_num=16 --reg=0.001
```
