# 官方更长图原生预算、顺序与核验指令边界（2026-10-03）

在任何新问题调用前冻结。延续第一目标：小模型迁移Trace as State并找边界；本阶段为4图探索，不是原256K规模完整复现，也不是前8短图结果的新确认。

## 样本与模型

官方修正版短图文件SHA54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516。按source顺序在[7000,9000)prompt_chars每类型取首2个图头SHA未出现在既有官方试点且彼此不同的行，交替Parents/BFS。不按gold或困难程度选择，原prompt/graph/Operation全部保留。字符档不是token档，和前短图的跨档差异不是控制难度的长度因果效应。

qwen3:1.7b本地digest8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7，实际metadata2.0B/Q4_K_M，最高ctx40960，本阶段统一ctx40960。温度0/top_p.9/think=true。官方tokenizer词条与本地GGUF编号151669对应一致；Ollama模板含/think，以前32响应API输入数全部校准一致。不声称render字节逐一验证，只验证token数和payload原文及运行窗口。

## 调用与因素

每图source8192、source16384各独立一次，先后顺序平衡；无重试/预热。唯一T来自source16384实际reasoning，不含visible content，不看gold、不修复错误；即使source失败/length/空仍保留该问题。固定serializer与50000字符头部上限。

六个reader构成3位置×2系统指令：pre、post、reread，以及check_pre、check_post、check_reread。pre/post保持相同T/x/q，最终Operation在末尾；reread没有T。check仅在system追加公开的程序核验要求：独立从实际图重新算、BFS用新visited、每节点核对实际父边或最短层数、忽略无依据的示例/轨迹答案。无新证据、程序执行、外部检索或gold。核验指令可能改善无T重读，因此三个位置都有对照，不把指令效果归因T或创新。

所有reader同8192输出cap，共8source+24reader=32请求。source16384与reader8192预算不同，source8192提供同预算单遍基线。source模型与reader同一模型，1trace/1repeat；与原frontier最大effort/无clientcap/5trace5repeat不同，末尾未重复附加原ANSWER_FORMAT。排序在冻结文件中记录，核验/不核验先后2:2平衡，不宣称准确墙钟速度。

## 输入资格与故障

每请求写入持久attempt账本，再发一次。保存完整request/raw响应/error body/token/termination，返回时先把response写到账本，再存结果矩阵。异常中断保留pending项，不重跑同矩阵覆盖未知原结果。提交前按已校准tokenizer计算输入+max output+64<=40960；不满足则不传输，计计划失败而非删掉原图/T。不裁原文、不动态缩T解决超窗。

返回逐次比较API prompt_eval_count与预计输入数，保存/api/ps运行窗口。资格要求计数相同、运行ctx至少配置值、原模型digest一致；不满足则不能解释成完整原文条件，但原可见评分保持，另给资格失效标记，不能改成成功。

## 评分与判读

只有正常stop的可见最后非空行严格Final Answer:[...]才评分；thinking不作答案。HTTP、length、missing/语法错误、提交前超窗计0；合法空集正常计算EM/setF1。全部图入分母，不筛source正确。

预先描述：source16384 vs8192有效率/F1/EM；普通pre-post与两臂vsreread；各check vs同位置普通臂；check_pre-check_reread用于区分额外T与指令。n4不做显著性/总体效应声明；逐图展示纠错（source16384错到reader正确）、新错（对到错）、错误答案继承。预设值得新图确认的候选：本阶段完整输入资格、全reader合法，F1与EM均不低于自身普通臂，至少2/4首轮错误图被纠正，且不以更多reader token换取该描述优势。这个候选门槛只选后续验证路线，不能称确认成功。

逐部署路径成本=同4source16384+对应4reader，另外报告同预算source8192和source16384单遍。source8192额外诊断支出也计阶段总支出，不必叠加到实际只用16384生产者的部署路径。prepare/tokenizer耗时另计；不称全项目端到端费用完整，历史中断未知支出另存。所有调用结束后再独立审计、R出图和范围模拟审稿；主论文暂不改写。
