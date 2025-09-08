### 引言
一个slide讲的很好[lec26_slides](https://www.cs.cornell.edu/courses/cs4756/2023sp/assets/slides_notes/lec26_slides.pdf)
系统而简洁的讲述了**为什么大语言模型来控制机器人是一条可行的路，以及大语言模型能够解决机器人控制里面的哪些难点和痛点**

首先提到现在控制机器人是非常rigid，就是说不够灵活，针对单一或者特定任务工程师进行调试编程，没有一种灵活的可以重编程的方式来适应各种任务。
那么**如何把工程师解放出来，只使用自然语言的指令来实现对机器人编程实现任务**？这个时候LLM就起到了关键性的作用。

-------
### Pre-LLM挑战
这个slide讲到了Pre-LLM(应用大模型钱)两个关键性的挑战：
- （Grounding）**如何把自然语言转换成能够被机器人识别、处理的状态**
-    (Planning) **如何规划合适的动作来完成任务**

#### Grounding
比如我想让机器人抓取位于桌子上最左边的红色方块，这里要解决定位到桌子、桌子上的物体、红色的方块以及相对位置。如果用逻辑语言来表述![[Pasted image 20250908104127.png]]
可以看到，其实这么简单的一句描述，想进行grounding也不是一件简单的事情。

#### Planning
任务规划对于机器人操作来说非常重要，其设计到如何对任务进行合理的拆解与执行。
比如下面这个例子，将货架上的苹果放到桌子上![[Pasted image 20250908104323.png]]

这个任务完成的流程图如下：
![[Pasted image 20250908104336.png]]
可以看到这个执行流程会随着任务的复杂程度不断增加。

----
### LLM Robotics
论文：**Do As I Can, Not As I Say: Grounding Language in Robotic Affordances**
Saycan**通过LLM对话，然后提供一组可选的动作列表，通过采用强化学习训练的value functions来选择当前最合适的动作。**![[Pasted image 20250908104613.png]]
Saycan为什么能有效处理Grounding和planning？
![[Pasted image 20250908104658.png]]
#### 问题
本质上是通过LLM的内在知识处理grounding，通过chain of thought 进行任务规划。
但是仍然存在问题：
![[Pasted image 20250908104822.png]]
##### question 1: 如何处理失败
论文：**Inner Monologue: Embodied Reasoning through Planning with Language Models**
Inner Monologue 内心独白，**通过让LLM构建一个闭环反馈系统，不断获取不同形式的反馈（场景描述，成功与否），来处理失败。**![[Pasted image 20250908105025.png]]
##### question2 : 如何判断正确性
论文：**Code as Policies: Language Model Programs for Embodied Control**
![[Pasted image 20250908105020.png]]
**通过迭代生成代码的方式，对之前生成的代码（比如函数未定义等错误）进行修改，可以生成更加复杂、可靠的执行代码。**

## 总结
这个slides主要讲了pre-LLM做机器人控制的两个难点和挑战（**grounding**, **planning**），然后介绍了SayCan能够借助**LLM**的内部知识来grounding以及Chain of Thoughts来完成planning。同时也介绍了LLM Robotics的两个问题，
如何处理**失败**以及验证**正确性**？
Inner Monologue通过构建获取不同形式的反馈构建闭环系统，来处理失败。
Code as Policies通过迭代生成代码对之前的代码进行修改与补充，生成更加复杂、可靠的代码。
