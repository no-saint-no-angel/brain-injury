from segmentation_file.calc_loss import diceCoeffv2


def calcu_accuracy(output, labels):
    # 打印预测图中，各个区域的分割的准确率
    calcu_accuracy = []
    BKG = diceCoeffv2(output[:, 0:1, :], labels[:, 0:1, :], activation=None).cpu().item()
    calcu_accuracy.append(BKG)
    injury = diceCoeffv2(output[:, 1:2, :], labels[:, 1:2, :], activation=None).cpu().item()
    calcu_accuracy.append(injury)

    mean_dice = (BKG + injury) / 2
    calcu_accuracy.append(mean_dice)
    return calcu_accuracy