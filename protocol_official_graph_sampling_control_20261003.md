# 原生思考采样配置对照（2026-10-03）

本阶段是在长图冻结实验进行时，核对官方Qwen3模型说明发现temperature0与思考模式建议不一致而提出。该发现是设置资格问题，尚不能断言此前失败因其引起。旧实验继续完成、不修改分数、限定为贪心配置结果；新阶段任何调用前冻结。

## 样本与对照

官方同修正版GraphWalks，[7000,9000)字符档按source顺序取每类型首2个图头SHA未出现于既有官方四阶段且彼此不同的行，4新图，Parents/BFS交替。不以答案选择。无T、无核验额外指令、原完整prompt与原graph system。锁同qwen3:1.7b digest/量化。

每图两个regime×两个seed202610037/202610038=4次，16请求。四组合轮转顺序，每组合在每位置一次。ctx32768（官方模型卡原生窗口），cap8192，think=true，无预热无重试。

greedy匹配控制temp0/top_p.9/top_k40；recommended模型官方基础建议temp.6/top_p.95/top_k20。两者共同min_p0/repeat_penalty1.0/相同seed。这是三个采样参数的整体regime比较，不是温度单因素，也不复刻旧未显式repeat_penalty默认。模型卡针对显著重复另建议presence_penalty1.5；本阶段未实现该附加建议，不能称全配置优化完成。不另装运行时、不重训练或换模型。

## 资格、评分与分析

写ahead完整请求账本、完整raw/usage/termination/error body、运行/api/ps窗口。逐次官方tokenizer输入数匹配API且输入+cap+64<=32768，runtime窗口/模型digest一致才具备完整输入资格。不称render字节验证，只校准token数与payload。保留任何HTTP/length/语法失败记0，不从thinking找答案。

每图两seed先平均，4图作为独立分析单位，不能把8repeat当8问题。描述每regime可见有效率、EM/setF1、token、终止原因、逐图和seed变化；n4不报显著性或总体收益。always-empty基线衡量答案集构成。后续值得检验条件：recommended完整输入、两个seed均有效，每图均F1不低于greedy、平均EM不低、至少2图平均F1更高、平均API总token不高。门槛只选继续路线，不能证明模型卡建议必然有效或顺序创新。

本阶段单遍total tokens=API input+output，两regime均无producer成本。阶段全部16调用成本单独存，不覆盖上一32调用或历史中断未知支出。两阶段数据不能拼成同分布/同配置长度效应。所有调用完成后再审计/R图/范围模拟审稿。若推荐配置仍差，先如实写边界，不继续同4图调参；必要下一阶段再检验原论文的5条源轨迹/重复数，不能用1条T否定5条T方法。
