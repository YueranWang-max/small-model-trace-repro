# 官方短图新图原生轨迹顺序重复（2026-10-03）

前2题探索显示qwen3:1.7b pre/post/reread平均setF1 .9444/.875/.375，但EM前后相同，不能确认效应。本协议在新问题任何调用之前冻结，用于追踪同T顺序线索。不是原论文完整256K复现或通用新方法。

同官方修正短图文件[2000,3500)prompt_chars，按source row顺序每类型首4个完整图头SHA未出现在前两接口试点且彼此不同的样本，8不同图。仅据类型/长度/图哈希选择，不看答案；交替Parents/BFS。模型tag qwen3:1.7b（本地元数据2.0B，Q4_K_M，锁定digest）；ctx32768，think=true，temp0，top_p.9，cap8192。每题source一次，三读出各一次，共32正式调用，无预热无重试。顺序遍历全部6种读出排列后复用前2种，位置支持2或3题，无严格性能比较。

source原prompt一次，取实际thinking唯一trace；visible content不入T。按固定原论文GraphWalks前言与first_run_reasoning_traces序列化，超50000字符截头。即使source错误/length/空/HTTP失败也保留该问题，T不修复，读出继续，失败入分母。pre完整T+原prompt，post原完整图头+相同T+最后Operation块，reread原prompt。system/最终q不变，各读出原生推理而非think=false。区别原论文n_tr=1/repeat1/短档/cap/temp，末尾未额外重复附加ANSWER_FORMAT。

评分原生可见最后非空行fullmatch Final Answer: [...]，正常stop才有效；不在thinking中找答案，length/HTTP/缺失语法计0。空集合法。主要F1配对对比pre-post、pre-reread；8图cluster bootstrap20000次seed202610035，双侧精确符号翻转2^8，两个主要比较Holm校正。推进门槛：两项F1差95%点态CI下界>0且Holm p<.05；pre EM不得低于post/reread；8source及24reader全部正常、合法终端答案。任何条件未通过均不称顺序收益得到确认，不换题不改提示。n8统计支持有限，即使通过也需更多新图/模型/长度档。

完整保存请求/原始响应/usage/错误体/源引用SHA，不筛选正确source。各部署路径总token=8source+各8reader；不得把producer省略。source同问题成功与否可作描述性分层，不能从主分析删掉失败。R图原始CSV可重建；审计/统计/绘图全部模型调用完成后执行。原先进程中断未知支出另列，不能声称全项目成本完整或普遍提速。若再意外中断，不直接覆盖未知请求，另立偏差记录。
