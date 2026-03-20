---
title: "The COMPAS Algorithm: A Case Study in AI Bias and Criminal Justice"
meta_title: ""
description: "A detailed examination of the COMPAS recidivism prediction algorithm, the ProPublica investigation that exposed racial bias in its predictions, and the broader lessons for AI fairness in high-stakes decision-making."
date: 2028-03-19
image: "/images/blogs/compas-bias/cover.jpg"
categories: ["AI Ethics"]
author: "Amar Singh"
tags: ["bias", "criminal-justice", "fairness", "case-study"]
draft: false
---

Few case studies in the history of artificial intelligence have done more to shape public understanding of algorithmic bias than the controversy surrounding COMPAS -- Correctional Offender Management Profiling for Alternative Sanctions. This proprietary risk assessment tool, developed by Northpointe (now Equivant), is used by courts across the United States to predict the likelihood that a criminal defendant will reoffend. The tool's influence is substantial: its risk scores inform decisions about bail, sentencing, and parole that directly affect people's freedom.

In 2016, the investigative journalism organization ProPublica published a landmark analysis alleging that COMPAS was biased against Black defendants. The investigation, and the fierce technical debate that followed, became a defining moment in the field of algorithmic fairness. It exposed fundamental questions about what fairness means when algorithms make high-stakes predictions about human behavior, and it demonstrated that these questions have no easy answers.

Understanding the COMPAS case in depth is essential for anyone working in AI, criminal justice, or public policy. It is not simply a story about a flawed algorithm. It is a story about the deep tension between different mathematical definitions of fairness, the challenge of building prediction systems in a society marked by systemic inequality, and the difficulty of holding proprietary algorithms accountable for their real-world impacts.

## What Is COMPAS?

COMPAS is a risk assessment instrument that generates scores intended to predict a defendant's risk of general recidivism (committing any new crime), violent recidivism (committing a new violent crime), and failure to appear in court. The tool combines information from a 137-question survey with criminal history data to produce risk scores on a scale of 1 to 10, where higher scores indicate higher predicted risk.

### How the Assessment Works

The COMPAS questionnaire collects information across multiple domains. Criminal history questions cover prior arrests, convictions, incarcerations, and juvenile offenses. Social and environmental questions ask about education, employment, housing stability, substance use, family relationships, and peer associations. Behavioral and attitudinal questions explore anger, criminal thinking patterns, and social adjustment.

Importantly, COMPAS does not directly use race as an input variable. The questionnaire does not ask about race, and race is not a feature in the algorithm's calculations. However, as we will see, the absence of race as an explicit input does not prevent the algorithm's outputs from exhibiting racial disparities.

### How Scores Are Used

COMPAS scores are used at multiple points in the criminal justice process. At the pretrial stage, judges may consider COMPAS scores when making bail or detention decisions. At sentencing, judges may use the scores to inform decisions about incarceration versus community supervision. In the correctional system, scores may influence decisions about programming, custody level, and parole eligibility.

The extent to which COMPAS scores influence judicial decisions varies significantly across jurisdictions. In some courts, scores are one of many factors considered by judges. In others, they carry considerable weight. The Wisconsin Supreme Court ruled that COMPAS scores can be considered at sentencing but cannot be the sole basis for a decision, and that defendants must be informed of the tool's limitations.

### The Scale of Deployment

COMPAS is one of the most widely used risk assessment instruments in the American criminal justice system. It has been deployed in jurisdictions across the country, from large urban courts to small rural ones. The scale of its deployment means that any biases in the system affect a large number of individuals and have systemic implications for criminal justice outcomes.

## The ProPublica Investigation

### Methodology

In 2016, ProPublica obtained COMPAS scores for over 7,000 defendants arrested in Broward County, Florida, between 2013 and 2014. They matched these scores with actual recidivism outcomes -- whether defendants were charged with new crimes within two years of their COMPAS assessment.

ProPublica's analysis focused on the accuracy and error patterns of COMPAS predictions, specifically examining whether the tool's errors were distributed equally across racial groups. Their methodology was straightforward: compare what COMPAS predicted (high risk or low risk) with what actually happened (did the person reoffend or not), and check whether the patterns of correct and incorrect predictions differed between Black and White defendants.


![Illustration representing fairness and bias in AI systems](/images/blogs/pool-ethics/3.jpg)

### Key Findings

ProPublica's analysis produced several findings that generated significant public attention and concern.

**False positive rate disparity**: Among defendants who did not reoffend within two years, Black defendants were roughly twice as likely as White defendants to have been classified as high risk by COMPAS. In other words, COMPAS was significantly more likely to incorrectly label a Black defendant as high risk -- a false positive. The false positive rate was approximately 45% for Black defendants versus approximately 23% for White defendants.

**False negative rate disparity**: Among defendants who did reoffend, White defendants were roughly twice as likely as Black defendants to have been classified as low risk. COMPAS was more likely to incorrectly label a White defendant as low risk -- a false negative. The false negative rate was approximately 48% for White defendants versus approximately 28% for Black defendants.

**Overall accuracy**: COMPAS correctly predicted recidivism about 61% of the time for both Black and White defendants. The overall accuracy rate was roughly similar across racial groups.

**Racial disparity in scores**: Black defendants received higher average COMPAS scores than White defendants, even after controlling for criminal history and other relevant factors.

ProPublica concluded that COMPAS was biased against Black defendants because it was more likely to wrongly flag them as future criminals and more likely to wrongly label White defendants as low risk.

## Northpointe's Response and the Fairness Debate

Northpointe vigorously contested ProPublica's characterization of COMPAS as biased. Their response, published in a detailed technical report, pointed to a different set of fairness criteria that COMPAS did satisfy -- and this disagreement became the foundation of one of the most important technical debates in the field of algorithmic fairness.

### Predictive Parity

Northpointe argued that COMPAS achieved predictive parity -- the principle that a given risk score should mean the same thing regardless of the defendant's race. Among defendants who received a risk score of 7, for example, roughly the same percentage of Black and White defendants actually went on to reoffend. In statistical terms, the positive predictive value (the probability of reoffending given a high risk score) was approximately equal across racial groups.

This is a meaningful form of fairness. From a decision-maker's perspective, predictive parity means that a COMPAS score of 7 carries the same information content whether the defendant is Black or White. A judge who sees a score of 7 can interpret it consistently without needing to adjust for the defendant's race.

### The Impossibility Result

The clash between ProPublica's analysis and Northpointe's response exposed a mathematical reality that has become one of the most important findings in fairness research: when base rates differ between groups, it is generally impossible to simultaneously achieve both equal false positive/negative rates and predictive parity.

In the Broward County data, the base rate of recidivism was higher among Black defendants than White defendants. This difference in base rates reflects a complex web of social, economic, and systemic factors -- poverty, policing patterns, historical discrimination, unequal access to education and employment -- but the statistical reality is that the rates differ.

Given this difference in base rates, a calibrated predictor (one that achieves predictive parity) will necessarily produce different false positive and false negative rates between the two groups. This is not a flaw in the algorithm -- it is a mathematical constraint. You can have equal false positive rates, or you can have predictive parity, but you generally cannot have both when the base rates differ.

This impossibility result, formalized independently by multiple research groups in 2016 and 2017, transformed the fairness debate. It showed that the disagreement between ProPublica and Northpointe was not about whose analysis was correct -- both were right about the fairness criteria they measured. The disagreement was about which fairness criteria should take priority, and that is a normative question, not a technical one.

## Deeper Technical Analysis

### What Does "Bias" Actually Mean Here?

The COMPAS controversy demonstrates that "bias" in algorithmic systems can mean very different things depending on the definition used.

**Statistical bias**: In the traditional statistical sense, a predictor is biased if it systematically over-predicts or under-predicts outcomes. COMPAS appears to be approximately unbiased in this sense -- its predictions are calibrated similarly for both racial groups.

**Disparate error rates**: ProPublica's definition of bias focuses on whether the algorithm's errors are distributed equitably across groups. Under this definition, COMPAS is biased because Black defendants bear a disproportionate share of false positive errors.

**Disparate impact**: A broader legal concept of bias asks whether the algorithm's decisions have a disproportionate adverse impact on a protected group. To the extent that higher COMPAS scores lead to harsher treatment (longer detention, less favorable bail terms, longer sentences), the fact that Black defendants receive higher average scores constitutes disparate impact.

**Individual fairness**: Yet another definition asks whether similar individuals receive similar scores, regardless of race. This definition raises the difficult question of what makes two defendants "similar" -- should similarity be based on criminal history alone, or should it account for the social circumstances that shape criminal behavior?

Each of these definitions captures a legitimate aspect of fairness, and each leads to a different assessment of whether COMPAS is biased. The lesson is that "algorithmic bias" is not a single, clear-cut concept but a family of related concerns that can point in different directions.

### The Role of Base Rates

The difference in base rates between racial groups is central to the COMPAS debate, but it also raises deeper questions. Why do base rates differ? If the difference reflects differential policing -- Black communities being policed more intensively, leading to higher arrest rates that do not reflect higher rates of criminal behavior -- then using historical arrest data to predict future arrests perpetuates and potentially amplifies existing biases in the criminal justice system.

Research suggests that policing patterns do vary significantly by race and geography, and that arrest rates are an imperfect proxy for criminal behavior. Drug use rates, for example, are roughly similar across racial groups, but arrest rates for drug offenses are dramatically higher for Black individuals. If COMPAS is trained on arrest data that reflects these policing disparities, its predictions will encode those disparities even without explicit racial inputs.

This creates a feedback loop. Biased policing produces biased arrest data. Algorithms trained on biased data produce biased predictions. Biased predictions lead to biased decisions (harsher bail terms, longer sentences). Biased decisions contribute to higher future arrest rates for the affected group. The cycle repeats.


![Visual depicting the ethical considerations of algorithmic decision-making](/images/blogs/pool-ethics/4.jpg)

### Proxy Variables

Although COMPAS does not use race as an input variable, many of the variables it does use are correlated with race due to systemic inequality. Neighborhood characteristics, employment history, educational attainment, and family structure are all associated with both race and criminal justice outcomes. When these variables are used as predictors, they can act as proxies for race, producing racially disparate outcomes even without explicit racial classification.

This proxy variable problem is one of the fundamental challenges in algorithmic fairness. Simply removing protected attributes (race, gender, age) from the input features does not prevent the algorithm from making predictions that are correlated with those attributes. The correlations are embedded in the training data through centuries of systemic inequality, and they cannot be eliminated by ignoring the protected attribute.

## The Broader Impact

### Legal and Policy Responses

The COMPAS controversy has prompted legal and policy responses at multiple levels. Several states have passed or considered legislation regulating the use of risk assessment tools in criminal justice. The debate has influenced how courts interpret due process and equal protection requirements when algorithms are used in sentencing. Defense attorneys have challenged COMPAS scores in individual cases, arguing that the proprietary nature of the algorithm prevents defendants from examining the evidence against them.

The Wisconsin Supreme Court's decision in State v. Loomis addressed some of these concerns, ruling that COMPAS scores could be used at sentencing but establishing several limitations: judges must be advised of the tool's limitations, including its potential for racial bias; COMPAS scores cannot be used to determine whether to incarcerate someone; and defendants must have the opportunity to challenge the accuracy and relevance of the scores.

### Academic Research

The COMPAS case catalyzed an enormous body of academic research on algorithmic fairness. Computer scientists, statisticians, legal scholars, and social scientists have produced hundreds of papers exploring different definitions of fairness, the mathematical relationships between them, and methods for achieving various fairness criteria. The impossibility results inspired by the COMPAS debate have become foundational concepts in the field.

This research has produced practical tools and frameworks for assessing algorithmic fairness, including open-source software libraries, audit methodologies, and design principles for fair ML systems. While the research has not resolved the fundamental normative questions about which fairness criteria should take priority, it has provided the conceptual and technical infrastructure for more rigorous fairness analysis.

### Public Awareness

Perhaps the most significant impact of the COMPAS controversy has been on public awareness of algorithmic bias. The ProPublica investigation was widely covered in mainstream media and brought the concept of algorithmic fairness to a broad audience. The case has become a standard reference point in discussions about AI ethics, appearing in policy debates, educational curricula, and public discourse about the role of algorithms in society.

This public awareness has created pressure on technology companies and government agencies to take algorithmic fairness seriously. Organizations that deploy algorithmic decision-making systems increasingly face public scrutiny and are expected to demonstrate that they have considered and addressed fairness concerns.

## Lessons for AI Practitioners

### Lesson 1: Fairness Is Not a Technical Problem with a Technical Solution

The COMPAS case demonstrates that algorithmic fairness involves fundamental value judgments that cannot be resolved through better mathematics or more sophisticated algorithms. Which fairness criteria should take priority is a normative question that depends on ethical principles, legal requirements, and the specific context in which the algorithm is deployed. AI practitioners must engage with these normative questions rather than treating fairness as a purely technical optimization problem.


![Conceptual image showing the balance between AI power and responsibility](/images/blogs/pool-ethics/5.jpg)

### Lesson 2: Historical Data Encodes Historical Injustice

Algorithms trained on historical data will reflect the biases and inequities present in that data. In the criminal justice context, this means that models trained on arrest and conviction data will encode the effects of differential policing, prosecution, and sentencing patterns. AI practitioners must critically evaluate their training data for sources of bias and consider whether historical patterns are appropriate bases for future predictions.

### Lesson 3: Removing Protected Attributes Is Not Sufficient

The "fairness through blindness" approach -- simply removing protected attributes from the input features -- does not prevent algorithmic bias. Proxy variables carry racial (and other protected) information through the model, producing disparate outcomes even without explicit use of protected attributes. Achieving fairness requires more active interventions, such as fairness constraints during model training, post-processing adjustments to model outputs, or fundamental rethinking of what the model is trying to predict.

### Lesson 4: Transparency and Accountability Matter

COMPAS is a proprietary system, and its internal workings are trade secrets. This opacity makes it difficult for defendants, advocates, researchers, and policymakers to scrutinize the tool's methodology and assess its fairness. The case makes a strong argument for transparency in algorithmic systems used for high-stakes decisions. At minimum, stakeholders should be able to understand what factors influence the algorithm's predictions and how those factors are weighted.

### Lesson 5: Context Determines What Fairness Means

The appropriate definition of fairness depends on the specific context and consequences of the algorithmic system. In criminal justice, where the consequences include incarceration, the relevant fairness considerations may differ from those in contexts like credit scoring or hiring. AI practitioners must understand the stakes and context of their specific application when designing and evaluating fairness properties.

### Lesson 6: Prediction and Decision Are Different Things

COMPAS produces risk predictions, but it is humans (judges) who make decisions based on those predictions. The fairness of the overall system depends on both the fairness of the predictions and how those predictions are used in decision-making. A well-calibrated prediction tool can still lead to unfair outcomes if judges interpret and apply the scores inconsistently or if institutional policies translate scores into decisions in biased ways.

### Lesson 7: The Question of Whether to Use AI at All

The COMPAS debate sometimes focuses so intensely on how to make the algorithm fair that it overlooks a more fundamental question: should algorithmic risk assessment be used in criminal justice at all? Proponents argue that even imperfect algorithms are more consistent and transparent than human judicial discretion, which is also subject to bias. Critics argue that reducing complex human situations to numerical risk scores is fundamentally dehumanizing and that the veneer of mathematical objectivity obscures the value judgments embedded in the system.

This question -- whether algorithmic decision-making is appropriate in a given context -- is one that AI practitioners should confront honestly rather than assuming that the technical capability to build a system justifies its deployment.

## The Ongoing Debate

The COMPAS controversy remains unresolved and continues to generate debate. New research periodically revisits the data and analysis, sometimes reaching different conclusions. Jurisdictions continue to adopt, modify, or abandon risk assessment tools based on their assessments of the evidence.

What is clear is that the COMPAS case fundamentally changed the conversation about AI and fairness. It demonstrated that algorithmic bias is not a hypothetical concern but a concrete reality with real consequences for real people. It showed that fairness in machine learning involves deep, sometimes irreconcilable tensions between competing values. And it established that building fair algorithmic systems requires not just technical skill but ethical reflection, stakeholder engagement, and democratic accountability.

For AI practitioners, the COMPAS case is not just a cautionary tale -- it is a call to approach their work with the awareness that the systems they build will affect people's lives, and that the responsibility for ensuring those effects are just and equitable cannot be delegated to the algorithm.
