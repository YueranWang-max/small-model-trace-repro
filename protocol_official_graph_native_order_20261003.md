# 原生自产轨迹同T顺序小试点（2026-10-03）

前一独立冻结接口扩展完成后，两题8192 cap均产生正常可见终端答案的模型才纳入；依据接口有效性，不以正确性筛选。当前只有qwen3:1.7b合格。复用扩展的2问题，属于自适应同题探索，不是新题确认。

T来自各模型各问题8192响应唯一的message.thinking，保留全体推理、错误猜想；可见message.content不入T，不修复状态，不用gold构造。按原论文GraphWalks前言与first_run_reasoning_traces格式序列化，一条trace，超过50000字符取头部再...。

三臂：reread原released prompt；pre完整T在released prompt前；post在released prompt最后Operation块前插入相同T。图x和最终操作q逐字相同，q在末尾；same system/temp0/top_p.9/ctx32768/think=true/cap8192。每臂每题1调用，当前模型6次。题1pre-reread-post，题2post-reread-pre，依赖生产者调用已在前扩展完成。无重试，完整保存请求/响应/推理/content/终止/usage；不在模型调用间评分之外做审计绘图。

主要描述pre-post EM/setF1，及两臂vs重读；原native可见评分、非法/length/HTTP计0。n=2不报推断显著性或泛化置信度。全部臂完整终止并观察到pre优于post和reread才是后续新题确认线索，不是确认成功；任一失败仍保存且不换题不改提示。状态可含推理内答案猜想，这是原方法定义，不能据此称无答案信息泄漏；无外部gold喂入。

区别原研究：官方短图不是256K；1source trace/1repeat而非5；temperature0/8192输出cap与原frontier最大effort不一致；source和reader未重复附加原论文ANSWER_FORMAT末尾指令（released prompt已有答案示例）；本机量化模型/运行时不同。只复现原生字段重放与sameT/x/q布局这一机制，不声称全面忠实复现。API producer+reader成本全部计入每部署路径；source已支出不能忽略，前16格中断未知成本另列。
