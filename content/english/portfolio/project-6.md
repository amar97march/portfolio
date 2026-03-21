---
title: "JVB Health & Wellness Platform"
date: 2022-06-10T09:00:00+05:30
draft: false
description: "A real-time health data platform syncing and analyzing live data from Garmin and Apple Watch APIs, scaling to support 20,000+ users in a single day."
tags: ["Python", "Django", "REST API", "Garmin API", "Apple Watch", "Real-time Data"]
categories: ["Web Apps"]
image: "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800&h=500&fit=crop"
company: "Norm Software"
role: "Software Engineer"
year: "2019-2020"
tech_stack: ["Python", "Django", "REST API", "PostgreSQL", "Garmin API", "Apple HealthKit", "Celery", "Redis"]
category: "backend"
featured: false
---

A high-throughput health data platform built at Norm Software that ingests, synchronizes, and analyzes real-time health metrics from wearable devices. The system integrates with Garmin and Apple Watch APIs to provide users with unified health dashboards, scaling to support over 20,000 active users in a single day.

## Overview

JVB Health and Wellness is a fitness and wellness platform that aggregates health data from multiple wearable device ecosystems into a single, coherent view. Users connect their Garmin watches, Apple Watches, and other fitness trackers, and the platform continuously syncs their activity, heart rate, sleep, and other biometric data. The unified dashboard provides insights, goal tracking, and wellness recommendations powered by the aggregated data.

## Challenge

Wearable fitness devices generate continuous streams of health data, but each manufacturer provides data through proprietary APIs with different formats, rate limits, authentication flows, and data models. Users who own multiple devices or switch between brands lose continuity in their health tracking. The platform needed to abstract away these differences, providing a seamless experience regardless of the underlying device ecosystem. The additional challenge was scale -- wellness challenges and corporate health programs could drive sudden spikes in concurrent users, with events attracting over 20,000 participants in a single day, all expecting their data to sync in near real-time.

## Solution

Engineered the back-end systems responsible for device API integration, data synchronization, and health metric analysis. The architecture uses a multi-layer approach: device-specific adapter modules handle the idiosyncrasies of each API (authentication, pagination, data format), a normalization layer transforms raw data into a unified health metric schema, and an analytics layer computes derived metrics, trends, and insights from the normalized data. Celery workers manage the continuous synchronization cycle, with Redis-backed queues handling burst traffic during peak events.

## Key Features

- **Multi-Device Data Sync**: Dedicated integration modules for Garmin Connect API and Apple HealthKit that handle OAuth authentication, incremental data sync, and conflict resolution when overlapping data exists from multiple devices. The sync engine supports both push (webhook) and pull (polling) models depending on the API capabilities.

- **Real-time Data Streaming**: Health metrics including heart rate, step count, calories burned, and sleep stages are synced with minimal latency. During active wellness challenges, the system processes updates at high frequency to maintain leaderboards and progress trackers.

- **Unified Health Schema**: A normalized data model that maps manufacturer-specific metrics to a common schema. This abstraction enables consistent analytics, reporting, and visualization regardless of the source device. The schema handles unit conversions, timezone normalization, and data quality validation.

- **Scalable Event Support**: The architecture is designed to handle traffic spikes from corporate wellness events. Horizontal scaling of Celery workers, connection pooling, and request batching enable the system to process data for 20,000+ concurrent users without degradation.

- **Health Analytics Engine**: Computed metrics including daily activity scores, sleep quality indices, recovery metrics, and trend analysis over configurable time windows. The analytics engine runs as background tasks, precomputing results for fast dashboard rendering.

- **Data Privacy and Compliance**: Health data handling follows strict privacy protocols with encryption at rest and in transit, user-controlled data sharing permissions, and data retention policies compliant with health data regulations.

## Technical Highlights

The Django back-end is structured as a collection of loosely coupled apps, with the device integration layer designed around the adapter pattern. Each device API (Garmin, Apple) is encapsulated in an adapter that implements a common interface for authentication, data retrieval, and error handling. This design allows adding new device integrations without modifying the core sync logic.

The data synchronization pipeline uses Celery with Redis as the message broker, supporting priority queues to ensure that active challenge participants receive faster sync cycles than background users. Database writes are batched to reduce connection overhead during peak events, and PostgreSQL partitioning by date keeps query performance consistent as the dataset grows.

Rate limit management is a critical component -- each device API enforces different rate limits, and with thousands of users syncing concurrently, the system implements per-user and per-API token bucket rate limiting with intelligent backoff and retry strategies.

## Impact

- Scaled to support **20,000+ active users in a single day** during peak wellness events
- Successfully integrated real-time data streams from **Garmin and Apple Watch APIs** into a unified platform
- Maintained sub-minute data sync latency even during high-concurrency events
- Enabled corporate wellness programs to run large-scale health challenges with reliable, real-time leaderboards
- Provided users with a single health dashboard eliminating the need to check multiple device-specific apps
