# 五条原生自产轨迹、五次读出重复的小模型迁移（2026-10-03）

在前48调用结束、采样对照审计完成后提出。原论文5trace/5repeat与之前单trace单repeat不同，不能用单trace结果否定完整多轨迹方案。本阶段更接近原方法重复数和接口，但仍是本机短图/有cap的探索，尚未复现256K。

官方修正文件[2000,3500)字符，每类型首2个图头SHA未出现在之前五个官方阶段且彼此不同的行，4新图交替Parents/BFS。只据来源顺序/类型/字符档/图头选择，不看gold。保留原prompt并按原附录C给所有source/reader一致附加ANSWER_FORMAT末尾指令。system、固定PD、trace标签与最后Operation插入边界同原定义。

同qwen3:1.7b digest/Q4_K_M/ctx32768，think=true。采用官方基础采样temp.6/top_p.95/top_k20/min_p0，repeat_penalty1固定，未实现另建议presence_penalty1.5，不能称全部最优。每图5source，seed202610040–44，cap4096；实际message.thinking按来源顺序形成同一T，visible content不入T，错误、length或空推理不修复，不筛掉source或问题；每条超过50000字符取头部。完整5T不按好坏选择。Tpre/post原文和最终问题逐字一致，reread无T。每臂5repeat，seed202610050–54跨臂配对，cap8192，顺序按冻结排列轮转。共4×(5+3×5)=80请求，无预热无重试。

原生可见评分不从thinking提取，非法/HTTP/length/missing计0，合法空集正常。原始请求/响应/使用量/终止/错误体及持久attempt账本完整保存。输入数必须匹配API、运行ctx及digest一致，输入+输出cap+64<=32768；不裁原文/T解决超窗。Source4K与reader8K预算不同，source均分不作为同预算第一遍效果的因果对照，主要比较pre/post/reread8K。

独立图先平均5reader repeat，再在4图上平均，不能把20repeat当20独立图。描述两主要对比pre-post、pre-reread的F1/EM、逐图差及seed变异；4图bootstrap只作点态描述、不做显著性或总体零差主张。后续线索门槛：完整输入资格、reader全正常、pre F1对两臂各高至少5pp、EM不低、两比较分别至少2/4图正向。通过只值得更多新图验证，仍不证明通用创新或原文可舍弃。

额外公开基线：Majority@5按每个source可见预测节点至少出现在3/5集合构成程序答案，源失败作空集合，程序合法空输出正常评分，另报有效source数；Oracle@5取source最高事后分，仅特权诊断，不部署。全空预测baseline也报。不得把oracle写成算法。

成本：生产者5call全计。每图source成本总和+对应reader5repeat平均支出为一次轨迹部署路径的期望API token，不把producer除5；同时报告完整重复实验路径生产者5+reader5总量。reread部署最低成本可只运行一遍8K reader、不需要producer；单source4K均值、Majority@5的5source总量另列。全部80call与准备花费计研究阶段支出；历史中断未知支出仍在，不称全项目费用完整。分析/审计/R图在全部调用结束后执行。禁止依据这4图反复调提示；主论文继续保留未完成状态。
