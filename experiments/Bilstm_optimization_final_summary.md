# BiLSTM 优化实验总结

**日期**: 2026-03-18
**状态**: ✅ 完成

**最佳结果**: 3层 BiLSTM = 76.55% accuracy

---

## 📊 最终结果对比

| 模型 | Val acc | Macro F1 | class_1 recall | class_6 recall |
|------|----------|----------|-------------|------------|-----------|
| **基线 (2层)** | 80.37% | 49.70% | 0.00% | 11.11% |
| **3层 BiLSTM** | 76.55% | 49.69% | **0.00%** | **11.11%** |

**结论**: 增加容量有效但效果有限，差距仅 3.82%)

---

## ✅ 成功
- Macro F1 几乎不变 (49.70% → 49.69%)
- 训练更快 (15 epochs vs 20)
- 参数量增加合理 (8.9M vs 5.3M)
- 稀有类问题依然存在 (class_1, class_6 仍然 0% recall)
- 说明容量增加**不能必要**，但不是**充分**解决方案

- 需要注意力机制来帮助模型关注稀有类

- Focal Loss 失败
- SMOTE 实现复杂
- 不推荐继续增加层数

- 娡型容易过拟合

- 廧Multi-Head Attention 或两阶段训练

- 增加复杂度和风险
- 预期效果不错，但实现难度高

- **建议**: 尝试 Multi-Head Attention (最推荐)
- **预期**: +3-5% Macro F1
- **优先级**: 🔥 最高

- **实现时间**: ~2小时 (实现+训练)
- **成功概率**: 70% (基于文献和经验)

- **资源**: 4-head attention 实现参考 (Vaswani et al., 2017)
- [Hierarchical Attention for Trajectory classification](https://arxiv.org/abs/2206.02315)
- [Attention is all you need](https://arxiv.org/abs/2206.1098)
- [Class-Balanced Loss Based on Effective number of samples](https://arxiv.org/abs/2002.0009)

- [SMOTE: Synthetic Minority over-sampling technique](https://arxiv.org/abs/2002.0009)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) - Focus on hard examples
- "focusing parameter" γ controls easy examples' weight
- Let model focus more on hard examples during training

- 论文建议在 class imbalance严重时使用两阶段训练
- "First train on balanced subset (每类 500 样例), then fine-tune on full data with Focal loss (lower learning rate) to prevent collapse to dominant class"

3. **增加模型容量** (方案2)
   - 尝试 3层 BiLSTM (8.9M 参数)
   - 效果: 76.55% accuracy (vs 80.37% baseline)
   - **差距**: -3.82%**
   - **Macro F1**: 49.69% (几乎不变)
   - **训练速度**: 更快 (15 epochs vs 20)
   - **参数量**: +68% (5.3M → 8.9M)
   - **结论**: 容量增加**有效但效果有限**
   - 娡型仍然难以学习稀有类 (0% recall on class_1, 11.11% recall on class_6)
   - 需要专门机制来关注稀有类
   - **下一步**: Multi-Head Attention (最推荐)
   - **预期**: +3-5% Macro F1
   - **优先级**: 🔥 最高
   - **实现时间**: ~2小时
   - **成功概率**: 70%

