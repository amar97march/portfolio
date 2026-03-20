---
title: "What's Next? The AI Topics I'm Most Excited to Explore"
meta_title: ""
description: "A forward-looking exploration of the most exciting and important AI topics on the horizon, from multimodal reasoning and world models to AI governance, neuromorphic computing, and the evolving relationship between humans and intelligent systems."
date: 2029-03-26
image: "/images/blogs/whats-next/cover.jpg"
categories: ["Reflections"]
author: "Amar Singh"
tags: ["future", "predictions", "roadmap", "community"]
draft: false
---

This is the final post in a 365-day journey of writing about artificial intelligence. Over the past year, I have explored hundreds of topics spanning the full breadth of the field: foundational algorithms, cutting-edge research, real-world applications, career guidance, ethical considerations, tool reviews, and everything in between. As I close this chapter, I find myself not at an endpoint but at a beginning. The more I have learned, the more I have discovered there is to learn. The frontier of AI research and practice is expanding faster than any individual can track, and the topics that excite me most today are different from the ones that excited me a year ago.

In this final post, I want to look forward rather than backward. These are the AI topics, research directions, and open questions that I am most excited to explore in the months and years ahead. They represent my honest assessment of where the most important and interesting work is happening, informed by a year of intensive study and writing.

## Multimodal Reasoning and Understanding

The convergence of language, vision, audio, and other modalities into unified AI systems is one of the most significant trends in the field. Models that can reason across modalities, understanding not just what is in an image or what a text says but the relationship between visual and textual information, represent a fundamental advance in machine intelligence.

What excites me most is not the current generation of multimodal models, which are impressive but still limited in their reasoning capabilities, but the next generation. The key challenge is moving from multimodal perception (being able to process multiple modalities) to multimodal reasoning (being able to draw inferences that require integrating information across modalities).

Consider the difference between a model that can describe what is in a photograph and a model that can look at a photograph of a factory floor and identify safety violations based on its understanding of OSHA regulations. The second task requires not just visual recognition but the integration of visual information with domain knowledge, spatial reasoning, and causal inference. This kind of deep multimodal reasoning is at the frontier of current research, and progress here would unlock an enormous range of practical applications.

I am particularly interested in how multimodal reasoning intersects with embodied AI. Robots that can see, hear, and manipulate objects need to reason across sensory modalities in real time, making decisions that account for the physics of the real world. The gap between current multimodal models and the requirements of embodied intelligence is vast, but the path forward is becoming clearer.

## World Models and Simulation

The concept of world models, internal representations that allow an AI system to simulate how the world works and predict the consequences of actions, is one of the most intellectually compelling ideas in modern AI. Humans constantly use mental simulation to plan, predict, and understand. We can imagine what will happen if we push a glass off a table, predict how traffic will flow on a different route, and simulate social interactions before they happen.

Building AI systems with comparable simulation capabilities would be transformative. A world model that understands physics could power more capable robots. A world model that understands social dynamics could enable more nuanced conversational AI. A world model that understands economic systems could improve forecasting and planning.


![Emerging AI research directions and frontier topics](/images/blogs/pool-learning/6.jpg)

Current research on world models is being pursued from multiple directions. Video prediction models learn to simulate visual dynamics from large video datasets. Physics-informed neural networks incorporate known physical laws into their predictions. Generative models are being adapted to serve as simulators for planning and decision-making.

The challenge is that the real world is staggeringly complex, and no model can simulate it perfectly. The question is not whether world models will be perfect but whether they can be good enough to be useful, and for which tasks the approximation is acceptable. I suspect that the answer will vary dramatically across domains, and mapping out this landscape of applicability is work I am eager to do.

## The Science of AI Evaluation

As AI systems become more capable and are deployed in higher-stakes settings, the question of how to evaluate them becomes increasingly critical and increasingly difficult. Current evaluation practices, primarily based on benchmark datasets and aggregate metrics, are insufficient for understanding the true capabilities and limitations of modern AI systems.

I am excited about several directions in evaluation research. Behavioral evaluation, which tests AI systems in naturalistic scenarios rather than on curated test sets, provides a more realistic picture of real-world performance. Adversarial evaluation, which systematically probes for failure modes and edge cases, reveals vulnerabilities that standard benchmarks miss. Longitudinal evaluation, which tracks how system performance changes over time as the data distribution shifts, is essential for understanding deployment reliability.

Perhaps most importantly, I am interested in evaluation frameworks that go beyond accuracy to assess properties like fairness, robustness, calibration, and interpretability. A model that achieves 95% accuracy but is systematically wrong for a specific demographic group, or that is confident in its incorrect predictions, or that degrades catastrophically under slight distribution shift, is not a good model, regardless of its aggregate performance.

The development of better evaluation methods is not glamorous, but it is essential. Without rigorous evaluation, we cannot make informed decisions about when and how to deploy AI systems, and we cannot hold developers and deployers accountable for the systems they build.

## Efficient and Sustainable AI

The computational cost of training and running large AI models has grown exponentially over the past decade. Training a frontier language model now costs tens of millions of dollars and consumes enough electricity to power a small town. This trajectory is economically unsustainable and environmentally irresponsible. The next wave of AI research must prioritize efficiency alongside capability.

Several directions excite me. Model compression techniques, including pruning, quantization, knowledge distillation, and low-rank factorization, can reduce the computational requirements of large models by orders of magnitude with minimal loss in capability. Sparse mixture-of-experts architectures activate only a fraction of the model's parameters for each input, achieving better scaling properties. Efficient attention mechanisms reduce the quadratic cost of the transformer architecture.

Beyond algorithmic efficiency, I am interested in the hardware dimension. Specialized AI accelerators, from GPU clusters to purpose-built chips like Google's TPUs and various startup offerings, are evolving rapidly. The co-design of algorithms and hardware, where each is optimized with the other in mind, promises significant efficiency gains.

The sustainability aspect is deeply personal. As someone who writes about the benefits of AI, I feel a responsibility to also address its costs. An AI system that improves drug discovery but requires a coal power plant to run has not clearly made the world better. Finding the right balance between capability and resource consumption is one of the defining challenges of the field.

## AI for Scientific Discovery

Of all the applications of AI, its potential to accelerate scientific discovery is the one that excites me most. The history of science is, in large part, a history of new tools enabling new observations and new understanding. The telescope revealed the cosmos. The microscope revealed the cell. Genomic sequencing revealed the code of life. AI has the potential to be the next great scientific instrument, not by observing nature directly but by finding patterns in data that are invisible to human perception.

AlphaFold's prediction of protein structures was a watershed moment, demonstrating that AI could solve a problem that had stymied scientists for decades. But protein folding is just one of thousands of problems in biology, chemistry, physics, and materials science where AI could make transformative contributions.

I am particularly excited about AI for materials discovery. The search for new materials with specific properties, such as better batteries, more efficient solar cells, stronger and lighter structural materials, or more effective catalysts, is traditionally slow and expensive. AI systems that can predict material properties from composition and structure, suggest promising candidates for synthesis, and even design novel materials from scratch could dramatically accelerate this process.


![Human-AI collaboration interface design concepts](/images/blogs/pool-learning/7.jpg)

Climate science is another domain where AI has enormous potential. Climate models are computationally expensive, limiting the resolution and number of scenarios that can be simulated. AI-based emulators can approximate the behavior of full climate models at a fraction of the computational cost, enabling more comprehensive exploration of climate scenarios and faster evaluation of mitigation strategies.

In drug discovery, the next frontier beyond property prediction is the integration of AI across the entire drug development pipeline, from target identification through lead optimization to clinical trial design. Each stage presents distinct challenges, and the greatest impact will come from systems that can reason across the entire pipeline rather than optimizing individual stages in isolation.

## AI Governance and Institutional Design

The technical aspects of AI are inseparable from the governance structures that shape how the technology is developed and deployed. The regulatory landscape is evolving rapidly, with the EU AI Act, emerging US frameworks, and various national strategies around the world creating a complex patchwork of rules that practitioners must navigate.

What interests me most is not the specific regulations currently in place, which will continue to evolve, but the underlying questions of institutional design. How should oversight bodies be structured to keep pace with rapid technological change? What role should technical standards play in regulatory frameworks? How can we balance innovation incentives with safety requirements? What mechanisms can ensure that the benefits of AI are distributed broadly rather than concentrated among a few actors?

These are not purely technical questions, but they have deep technical dimensions. Effective AI governance requires policymakers who understand the technology and technologists who understand policy. The gap between these communities is one of the most important problems to solve, and I believe that technically grounded writing about governance issues can help bridge it.

I am also interested in the governance of AI within organizations. How should companies structure their AI development processes to ensure responsible practices? What internal review mechanisms are most effective? How can organizations balance the speed of development with the thoroughness of safety testing? These questions of organizational design are less discussed than national regulation but are at least as important for determining how AI is actually developed and deployed.

## Neurosymbolic AI and Reasoning

The limitations of purely neural approaches to AI, including their opacity, brittleness, and difficulty with systematic reasoning, have motivated a growing interest in neurosymbolic approaches that combine neural networks with symbolic reasoning systems.

The core insight is that neural networks and symbolic systems have complementary strengths. Neural networks excel at pattern recognition, handling noisy and ambiguous data, and learning from examples. Symbolic systems excel at logical reasoning, planning, and providing interpretable explanations. A system that combines both could potentially achieve capabilities beyond what either approach can deliver alone.

Recent progress in this area has been encouraging. Large language models have shown unexpected emergent reasoning capabilities that blur the traditional boundary between neural and symbolic computation. Chain-of-thought prompting and tool use allow language models to perform multi-step reasoning by generating intermediate steps. Neuro-symbolic architectures that use neural networks for perception and symbolic systems for reasoning have shown strong results in tasks requiring compositional generalization.

The question I find most interesting is whether we need explicit symbolic machinery or whether sufficiently large and well-trained neural networks can learn to perform symbolic reasoning implicitly. This is not just a technical question; it gets to the heart of what reasoning is and how it should be implemented in artificial systems.

## Human-AI Collaboration and Interaction

As AI systems become more capable, the nature of human-AI interaction is shifting from simple tool use to genuine collaboration. This shift raises profound questions about interface design, trust calibration, and the division of cognitive labor between humans and machines.

I am interested in how to design AI systems that complement rather than replace human capabilities. The most effective human-AI collaborations leverage the strengths of both parties: the AI's ability to process large amounts of data quickly and consistently, and the human's ability to apply judgment, context, and values to decisions.

Trust calibration is a particularly important challenge. Users who over-trust AI systems may accept incorrect outputs without verification. Users who under-trust AI systems may fail to benefit from accurate outputs. The ideal is calibrated trust, where users rely on the AI when it is likely to be correct and verify or override when it is likely to be wrong. Designing systems that communicate their uncertainty effectively and help users develop accurate mental models of the system's capabilities is an active area of research with enormous practical importance.

![AI for scientific discovery across multiple disciplines](/images/blogs/pool-learning/8.jpg)


I am also fascinated by the cognitive effects of working with AI systems over extended periods. How does regular use of AI assistance change how people think, learn, and make decisions? Are there skills that atrophy when AI takes over certain cognitive tasks? Are there new skills that emerge from human-AI collaboration? These questions sit at the intersection of AI research, cognitive science, and human factors engineering, and they will become increasingly important as AI becomes more deeply integrated into professional and personal life.

## Privacy-Preserving AI

The tension between AI's appetite for data and the growing demand for privacy protection is one of the most important challenges in the field. Techniques like federated learning, differential privacy, homomorphic encryption, and secure multi-party computation offer paths forward, but each comes with trade-offs in terms of model quality, computational cost, and implementation complexity.

I am excited about the maturation of federated learning in particular. The idea of training models on distributed data without centralizing it is compelling for healthcare, finance, and other domains where data cannot easily be shared. Recent advances in federated optimization, communication efficiency, and handling of non-IID data distributions are making federated learning increasingly practical.

Differential privacy, which provides mathematical guarantees about the privacy of individual data points, is another area where I see significant potential. The challenge is that strong privacy guarantees typically come at the cost of model utility, and finding the right balance for each application requires careful analysis.

The intersection of privacy-preserving AI with regulation is also important. Privacy regulations like GDPR create legal requirements that technical privacy mechanisms can help satisfy. Understanding how these technical and legal frameworks interact is essential for practitioners building AI systems that handle personal data.

## The Path Forward

Looking at this list of topics, I am struck by how different it is from the list I would have written a year ago. Before this project, my interests were more narrowly focused on the technical core of machine learning. Now, after exploring the full breadth of the field, I am drawn to the topics that sit at the intersection of technical innovation and real-world impact.

The AI field is at a fascinating inflection point. The foundational capabilities, large-scale pre-training, transfer learning, multimodal processing, and increasingly capable reasoning, are maturing rapidly. The challenge is shifting from "can we build AI that works?" to "how do we build AI that works reliably, safely, efficiently, and equitably?" This shift requires not just better algorithms but better evaluation methods, better governance structures, better interfaces, and better integration with the domains where AI is deployed.

I do not plan to write daily about AI going forward, at least not immediately. But I will continue writing, exploring these topics in depth, sharing what I learn, and engaging with the community that has formed around this project. The 365-day commitment ends tomorrow, but the learning continues.

## A Final Thank You

To everyone who has read, shared, commented on, disagreed with, corrected, or been inspired by any of the 365 posts in this series: thank you. Writing is often described as a solitary activity, but this project has been anything but. Every piece of feedback, every question, every suggested topic, and every shared experience enriched the work and made it more valuable than I could have produced alone.

The AI community is remarkable in its generosity with knowledge and its willingness to engage with new ideas. I am privileged to be part of it, and I look forward to continuing the conversation in whatever form it takes next.

The future of AI is not something that happens to us. It is something we build, collectively, through our research, our engineering, our writing, our policy work, and our daily choices about how to develop and deploy these powerful technologies. The topics I have outlined in this post are the ones I believe matter most for shaping that future positively. I hope they inspire you to explore, question, and contribute in your own way.

Here is to the next 365 days, and to whatever comes after.
