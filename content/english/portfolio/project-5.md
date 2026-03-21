---
title: "Cyberpunk 2077 Audio Tool"
date: 2023-06-10T09:00:00+05:30
draft: false
description: "A mission-critical studio management tool for the AAA game Cyberpunk 2077, managing the complete voice-over workflow from talent hiring to audio asset tracking."
tags: ["Vue.js", "Python", "Django", "REST API", "Audio Processing", "Game Development"]
categories: ["Web Apps"]
image: "https://images.unsplash.com/photo-1535223289827-42f1e9919769?w=800&h=500&fit=crop"
company: "Pole to Win (now Side)"
role: "Software Engineer"
year: "2020-2021"
tech_stack: ["Vue.js", "Python", "Django", "REST API", "PostgreSQL", "Audio Processing"]
category: "web-apps"
featured: true
---

A mission-critical studio management platform developed at Pole to Win for CD Projekt Red's AAA title Cyberpunk 2077. The tool manages the entire voice-over production pipeline -- from casting and hiring voice talent to recording session scheduling and tracking thousands of character audio assets through to final delivery.

## Overview

Cyberpunk 2077 features an expansive open world with hundreds of characters, each requiring professional voice acting across multiple languages. Managing a production of this scale demands a purpose-built platform that coordinates voice talent, recording studios, audio engineers, and production managers across a complex, interdependent workflow. This tool serves as the central hub for the entire audio production lifecycle, ensuring that every line of dialogue is cast, recorded, reviewed, and delivered on schedule.

## Challenge

AAA game audio production involves an enormous volume of assets -- Cyberpunk 2077 features thousands of individual voice lines across its main storyline, side quests, ambient dialogue, and environmental audio. Tracking which lines have been cast, recorded, reviewed, and approved across multiple characters, actors, languages, and recording sessions is a logistical challenge that quickly outgrows spreadsheet-based tracking. The production team needed a centralized system that provides real-time visibility into production status, identifies bottlenecks, and ensures nothing falls through the cracks during a multi-year development cycle.

## Solution

Developed core features for the studio management platform using Vue.js on the front-end and Django REST APIs on the back-end. The application models the complete voice-over workflow as a state machine, tracking each audio asset from initial script import through casting, scheduling, recording, review, and final delivery. The system provides different views optimized for each role -- casting directors see talent availability and audition status, recording engineers see session schedules and booth assignments, and production managers see aggregate progress dashboards.

## Key Features

- **Talent Management Module**: A comprehensive database of voice actors with profiles including language capabilities, character range, availability windows, and rate information. The casting workflow supports audition submission, review, and selection with multi-stakeholder approval chains.

- **Script and Dialogue Management**: Scripts are imported and broken down into individual dialogue lines, each tagged with character, scene, emotional context, and technical requirements. Lines are grouped into recording sessions based on character and actor availability.

- **Recording Session Scheduler**: A calendar-based scheduling system that coordinates studio availability, actor bookings, and engineer assignments. The scheduler accounts for dependencies -- ensuring prerequisite scenes are recorded before reactions, and that actors have context for emotional continuity.

- **Audio Asset Tracker**: Every recorded line is tracked through a multi-stage review pipeline: raw recording, edited take selection, quality assurance, director approval, and final mastering. Assets are versioned, and the system maintains a complete history of takes, notes, and approvals.

- **Production Dashboard**: Real-time progress visualization showing completion percentages by character, quest line, language, and overall project. The dashboard highlights at-risk items where recording or review deadlines are approaching with incomplete assets.

- **Localization Workflow**: The platform supports multi-language productions, tracking parallel recording efforts across localization teams. Each language maintains its own casting, recording, and review pipeline while sharing the same source script structure.

## Technical Highlights

The Vue.js front-end is built around a flexible data grid component capable of displaying and filtering thousands of dialogue lines with responsive performance. The component supports inline editing, bulk status updates, and configurable column layouts tailored to each user role. Complex filtering allows production managers to quickly surface specific subsets -- for example, all unrecorded lines for a particular character in the German localization.

The Django back-end implements a state machine pattern for asset lifecycle management. Each audio asset transitions through defined states with validation rules ensuring proper workflow adherence -- a line cannot be marked as approved without passing through quality assurance, and re-recording requests automatically reset downstream approvals. The REST API supports both granular single-asset operations and bulk actions for efficient batch processing during recording sessions.

The audio file management system handles large binary assets efficiently, with chunked uploads, background processing for format conversion and metadata extraction, and integration with the studio's existing file storage infrastructure.

## Impact

- Managed the complete voice-over production pipeline for one of the **most anticipated AAA game releases** in the industry
- Tracked **thousands of individual audio assets** across characters, quests, and multiple languages
- Provided real-time production visibility that enabled proactive identification and resolution of scheduling bottlenecks
- Streamlined the casting-to-delivery workflow, reducing coordination overhead between distributed production teams
- Supported the multi-language localization effort by maintaining parallel production pipelines with shared source material
