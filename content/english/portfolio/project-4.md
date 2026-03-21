---
title: "Workforce Management Platform"
date: 2024-09-20T09:00:00+05:30
draft: false
description: "A large-scale employee management platform supporting 4,000+ employees with automated workflows for leave tracking, attendance, and salary calculations."
tags: ["Vue.js", "Nuxt.js", "Python", "Django", "REST API", "PostgreSQL"]
categories: ["Web Apps"]
image: "https://images.unsplash.com/photo-1551434678-e076c223a692?w=800&h=500&fit=crop"
company: "Pole to Win (now Side)"
role: "Software Engineer"
year: "2020-2021"
tech_stack: ["Vue.js", "Nuxt.js", "Python", "Django", "REST API", "PostgreSQL", "Celery"]
category: "web-apps"
featured: true
---

A comprehensive employee management platform built at Pole to Win (now Side), supporting over 4,000 employees across multiple offices. The system automates critical HR workflows including leave tracking, attendance management, and salary calculations, saving more than 20 administrative hours per month.

## Overview

Pole to Win is a global services company with thousands of employees distributed across offices worldwide. Managing HR operations at this scale required a centralized platform capable of handling diverse employment types, regional labor regulations, and complex payroll calculations. This platform replaced a patchwork of spreadsheets and manual processes with an integrated, automated system that HR teams and employees interact with daily.

## Challenge

The existing HR workflow relied heavily on manual spreadsheet management and email-based approval chains. Leave requests were tracked in shared spreadsheets prone to conflicts and errors. Attendance was recorded through disparate systems with no centralized view. Salary calculations, factoring in overtime, leave balances, regional tax rules, and benefits deductions, consumed days of manual effort each pay cycle. With a workforce exceeding 4,000 employees, even small inefficiencies multiplied into significant operational costs and frequent payroll errors that damaged employee trust.

## Solution

Owned the development of a full-featured workforce management platform. The front-end, built with Vue.js and Nuxt.js, provides role-specific dashboards for employees, managers, and HR administrators. The Django back-end handles complex business logic for leave accrual, attendance tracking, and payroll calculations through a well-structured REST API. Celery workers manage asynchronous tasks such as payroll batch processing, report generation, and notification delivery.

## Key Features

- **Leave Management System**: A complete leave lifecycle management module supporting multiple leave types (annual, sick, personal, maternity/paternity), configurable accrual policies, carry-forward rules, and multi-level approval workflows. Employees submit requests through an intuitive calendar interface, while managers review and approve through a streamlined queue.

- **Attendance Tracking**: Real-time attendance recording with support for multiple check-in methods. The system handles edge cases like split shifts, overtime tracking, and grace periods. Automated absence notifications alert managers to unexpected no-shows.

- **Automated Salary Calculation**: A rules-based payroll engine that computes gross and net salary factoring in attendance records, leave deductions, overtime premiums, tax withholdings, and benefits. The calculation engine supports regional variations and is configurable per employment contract type.

- **Manager Dashboard**: A comprehensive view showing team attendance status, pending leave approvals, headcount summaries, and upcoming schedule conflicts. Managers can drill down into individual employee records and generate team-level reports.

- **Employee Self-Service Portal**: Employees access their attendance history, leave balances, payslips, and personal information through a responsive web interface. The portal reduces HR inquiry volume by enabling employees to find answers independently.

- **Reporting and Analytics**: Configurable report templates for HR teams covering attendance patterns, leave utilization, department headcounts, and payroll summaries. Reports can be generated on-demand or scheduled for automated delivery.

## Technical Highlights

The Vue.js front-end employs a modular component architecture, with shared UI primitives and feature-specific modules for each HR domain. Nuxt.js provides server-side rendering for fast initial page loads, particularly important for employees accessing the portal on varied network conditions across global offices. The calendar and scheduling components handle complex date logic including regional holidays, custom work weeks, and timezone-aware display.

The Django REST back-end is structured around domain-driven modules -- leave management, attendance, payroll, and user management each operate as independent Django apps with clearly defined API boundaries. The payroll calculation engine is designed as a pipeline of composable rules, making it straightforward to add new deduction types or regional tax calculations without modifying existing logic. Celery handles batch operations, processing payroll for thousands of employees in parallel while providing progress tracking to the administrator interface.

Database design prioritizes audit compliance -- every record change is versioned, and payroll calculations maintain a complete computation trace showing how each figure was derived.

## Impact

- Successfully supports **4,000+ active employees** across multiple offices and regions
- Saves over **20 administrative hours per month** through automated leave and attendance workflows
- Reduced payroll processing time from multiple days to hours through automated salary calculations
- Eliminated spreadsheet-based tracking, reducing data entry errors and conflicting records
- Earned recognition as **Top Performer Q2 2020** at Pole to Win
