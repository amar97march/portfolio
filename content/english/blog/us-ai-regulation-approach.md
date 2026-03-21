---
title: "The US Approach to AI Regulation: Executive Orders and NIST"
date: 2028-03-15T10:00:00+05:30
draft: false
description: "An analysis of the United States' approach to AI regulation, including executive orders, the NIST AI Risk Management Framework, state-level legislation, and the ongoing debate between innovation and oversight."
tags: ["AI Regulation", "US Policy", "NIST", "Executive Orders", "AI Governance", "Technology Law"]
categories: ["AI Regulation"]
image: "https://picsum.photos/seed/us-ai-regulation-approach-cover/1200/630"
keywords: ["US AI regulation", "NIST AI framework", "executive order AI", "American AI policy", "AI governance USA", "state AI laws"]
---

While the European Union has taken the path of comprehensive legislation, the United States has charted a fundamentally different course. The American approach to AI regulation is best described as **fragmented, evolving, and deliberately innovation-friendly**.

There is no single federal AI law in the United States. Instead, AI governance is distributed across executive orders, agency guidance, voluntary industry commitments, and a growing patchwork of state-level legislation. Understanding this landscape is essential for anyone building or deploying AI systems in or for the American market.

---

### The American Philosophy: Why No Comprehensive AI Law?

To understand the US approach, you must understand the underlying philosophy. Two powerful forces shape American technology policy:

**1. Innovation Supremacy**: There is a deep-seated belief in the US policy establishment that technological leadership is both an economic necessity and a national security imperative. Heavy-handed regulation, the argument goes, would drive AI talent and investment to other countries—particularly China.

**2. Sectoral Regulation Tradition**: Unlike Europe, which tends toward horizontal, cross-sector regulation, the US has historically regulated industries vertically. Financial services have the SEC and FDIC. Healthcare has the FDA. Transportation has the NHTSA. This tradition means AI regulation naturally falls to existing agencies within their domains rather than to a single comprehensive framework.

The result is a regulatory landscape that is both more flexible and more chaotic than Europe's.

---

![The US approach to AI regulation and governance](https://picsum.photos/seed/us-ai-regulation-approach-1/800/450)

### The NIST AI Risk Management Framework

The **National Institute of Standards and Technology (NIST)** released its AI Risk Management Framework (AI RMF 1.0) in January 2023. This is arguably the most influential document in US AI governance, even though it is entirely voluntary.

The framework is organized around four core functions:

**1. GOVERN**: Establish organizational structures and processes for AI risk management.
- Define roles and responsibilities
- Create policies and procedures
- Foster a culture of responsible AI development
- Engage diverse stakeholders

**2. MAP**: Contextualize the AI system within its operational environment.
- Identify intended and potential unintended uses
- Assess the broader societal context
- Map the system's potential impacts on individuals and communities
- Identify relevant legal and regulatory requirements

**3. MEASURE**: Quantify and assess AI risks.
- Evaluate system performance across diverse conditions
- Test for bias and fairness
- Assess reliability and robustness
- Monitor for emerging risks

**4. MANAGE**: Prioritize and act on identified risks.
- Implement risk mitigation strategies
- Establish monitoring and response plans
- Document decisions and trade-offs
- Plan for system decommissioning

The framework is deliberately flexible. It does not prescribe specific technical requirements but provides a structured approach for organizations to think through AI risks systematically.

```python
# Simplified representation of NIST AI RMF core functions
class NISTAIRiskFramework:
    """
    A conceptual implementation of NIST AI Risk Management Framework.
    The four core functions: Govern, Map, Measure, Manage.
    """

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.governance = {}
        self.risk_map = {}
        self.measurements = {}
        self.management_actions = {}

    def govern(self, policies: list, roles: dict, culture_assessment: str):
        """Establish AI governance structures."""
        self.governance = {
            "policies": policies,
            "roles_and_responsibilities": roles,
            "organizational_culture": culture_assessment,
            "stakeholder_engagement": [],
        }
        return self.governance

    def map_risks(self, intended_use: str, context: dict,
                  potential_impacts: list):
        """Contextualize AI system and identify risks."""
        self.risk_map = {
            "intended_use": intended_use,
            "operational_context": context,
            "potential_impacts": potential_impacts,
            "unintended_uses": [],
            "affected_stakeholders": [],
        }
        return self.risk_map

    def measure(self, metrics: dict, bias_results: dict,
                robustness_score: float):
        """Quantify and assess identified risks."""
        self.measurements = {
            "performance_metrics": metrics,
            "bias_assessment": bias_results,
            "robustness_score": robustness_score,
            "reliability_tests": {},
        }
        return self.measurements

    def manage(self, mitigations: list, monitoring_plan: dict):
        """Act on and monitor risks."""
        self.management_actions = {
            "risk_mitigations": mitigations,
            "monitoring_plan": monitoring_plan,
            "incident_response": {},
            "documentation": {},
        }
        return self.management_actions
```

---

### Executive Orders: The President's Tool

In the absence of comprehensive legislation, executive orders have become the primary federal tool for AI governance.

**The October 2023 Executive Order on Safe, Secure, and Trustworthy AI** was the most significant. Its key provisions included:

- **Safety Testing**: Companies developing foundation models that exceed certain compute thresholds must share safety test results with the federal government before release.
- **Red-Teaming Standards**: NIST was directed to develop standards for red-teaming (adversarial testing) of AI systems.
- **Watermarking**: The Commerce Department was tasked with developing guidelines for watermarking AI-generated content.
- **Bias and Discrimination**: Federal agencies were directed to address AI-related civil rights violations and create safeguards against algorithmic discrimination.
- **Privacy**: The executive order called for federal support of privacy-preserving technologies and data protection research.
- **Workforce Impact**: Agencies were directed to study AI's impact on the labor market and develop strategies for workforce transition.

However, executive orders have a fundamental limitation: **they can be reversed by the next president**. They do not carry the permanence of legislation, and their enforcement depends on the priorities of the current administration.

---

### Agency-Level Regulation

Several federal agencies have taken independent action on AI within their domains:

**The Federal Trade Commission (FTC)** has been the most aggressive regulator, using existing consumer protection and anti-discrimination laws to target AI harms. The FTC has taken enforcement actions against companies for:
- Deceptive AI marketing claims
- Biased AI-driven decision-making
- Unfair data collection practices for AI training

**The Equal Employment Opportunity Commission (EEOC)** has issued guidance on how existing anti-discrimination laws (Title VII, the ADA) apply to AI-powered hiring tools.

**The Food and Drug Administration (FDA)** has developed a framework for regulating AI and machine learning in medical devices, including requirements for ongoing monitoring and updates.

**The Securities and Exchange Commission (SEC)** has proposed rules addressing AI-related conflicts of interest in financial services.

**The Department of Defense (DoD)** has adopted ethical principles for AI use in defense and established the Chief Digital and Artificial Intelligence Office (CDAO) to coordinate military AI development.

---

![Federal agencies and executive orders shaping AI policy](https://picsum.photos/seed/us-ai-regulation-approach-2/800/450)

### State-Level Legislation: The Patchwork

In the absence of federal legislation, states have stepped in with their own AI laws. This has created a complex patchwork that companies must navigate:

**Colorado** passed the first comprehensive state AI law in 2024, regulating "high-risk AI systems" that make "consequential decisions" affecting consumers in areas like employment, education, financial services, and healthcare.

**Illinois** has the Biometric Information Privacy Act (BIPA), which has been used extensively in litigation related to facial recognition and biometric AI.

**California** has multiple AI-related bills addressing deepfakes, AI transparency, and automated decision-making.

**New York City** implemented Local Law 144, requiring bias audits of automated employment decision tools.

**Texas, Virginia, and Connecticut** have passed various AI transparency and accountability measures.

This state-by-state approach creates compliance challenges similar to what companies face with varying data privacy laws across states.

---

### Voluntary Industry Commitments

A distinctive feature of the US approach has been reliance on voluntary commitments from industry. In July 2023, the White House secured voluntary pledges from leading AI companies including:

- **Safety Testing**: Commit to internal and external security testing of AI systems before release.
- **Information Sharing**: Share information about AI safety risks with government and other organizations.
- **Watermarking**: Develop technical mechanisms to identify AI-generated content.
- **Security**: Invest in cybersecurity and insider threat safeguards.
- **Research**: Prioritize research on societal risks of AI.

The effectiveness of voluntary commitments is hotly debated. Proponents argue they move faster than legislation and allow flexibility. Critics point out that commitments are not legally binding, lack enforcement mechanisms, and can be abandoned when they become inconvenient.

---

### The Congressional Landscape

Congress has introduced numerous AI-related bills, though comprehensive legislation has been elusive. Notable proposals include:

- Algorithmic accountability acts requiring impact assessments
- AI transparency requirements for federal agencies
- Proposals for a new federal AI regulatory agency
- Bills addressing AI in specific sectors (healthcare, education, criminal justice)

The challenge in Congress is the same tension that pervades the entire US approach: the desire to promote AI innovation versus the need to protect against AI harms. Neither party has reached internal consensus on the right balance, and AI regulation does not split neatly along partisan lines.

---

### Comparing the US and EU Approaches

| Dimension | United States | European Union |
|-----------|--------------|----------------|
| **Structure** | Fragmented, sectoral | Comprehensive, horizontal |
| **Legal Basis** | Executive orders, agency guidance, state laws | Single legislative act |
| **Philosophy** | Innovation-first | Rights-first |
| **Enforcement** | Varies by agency | Centralized + national authorities |
| **Scope** | Voluntary + targeted mandates | Mandatory across risk tiers |
| **Flexibility** | High | Lower |
| **Predictability** | Lower | Higher |

---

![Navigating the patchwork of state and federal AI laws](https://picsum.photos/seed/us-ai-regulation-approach-3/800/450)

### What This Means for Developers

If you are building AI for the US market:

1. **Monitor state laws.** The compliance landscape varies by state and is changing rapidly. If you operate nationally, you need to track legislation in multiple states.

2. **Use NIST as your baseline.** Even though it is voluntary, the NIST AI RMF is the closest thing to a federal standard. Adopting it demonstrates due diligence and prepares you for future mandatory requirements.

3. **Watch the FTC.** The FTC has shown it will use existing authority aggressively. Ensure your AI claims are accurate, your systems are tested for bias, and your data practices are defensible.

4. **Prepare for convergence.** The current fragmented approach is likely temporary. Comprehensive federal legislation will eventually come, and it will probably draw heavily on NIST frameworks and existing agency guidance.

5. **Consider EU compliance anyway.** If there is any chance your system will be used by European residents, building to EU AI Act standards saves you from maintaining two different compliance postures.

---

### My Assessment

The US approach has both strengths and weaknesses. Its flexibility allows rapid adaptation and avoids the risk of locking in requirements that may be outdated by the time they are enforced. But the lack of a comprehensive framework creates uncertainty, compliance complexity, and gaps in protection.

The biggest risk is not that the US will over-regulate AI. It is that the patchwork approach will create enough confusion and inconsistency that companies effectively self-regulate—and as history shows, self-regulation in technology tends to serve corporate interests more than public ones.

---

*This is Day 255 of my AI blog series. Next, we look at AI audits—what they are, how they work, and why they are becoming essential.*
