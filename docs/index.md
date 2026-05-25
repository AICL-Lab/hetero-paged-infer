---
layout: home
title: Hetero-Paged-Infer
---

## Hetero-Paged-Infer

<div class="language-select">
  <a class="language-card" href="./en/">
    <div class="language-name">English</div>
    <div class="language-desc">Architecture, setup, API, deployment, and development guides</div>
  </a>
  <a class="language-card" href="./zh/">
    <div class="language-name">简体中文</div>
    <div class="language-desc">架构、安装配置、API、部署与开发指南</div>
  </a>
</div>

<style>
.language-select {
  display: flex;
  gap: 24px;
  justify-content: center;
  align-items: stretch;
  min-height: 60vh;
  padding: 48px;
}

.language-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 280px;
  padding: 32px 40px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  text-decoration: none;
  transition: all 0.3s ease;
}

.language-card:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand-1);
  box-shadow: var(--vp-shadow-3);
}

.language-name {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.language-desc {
  font-size: 0.95rem;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}

@media (max-width: 639px) {
  .language-select {
    flex-direction: column;
    min-height: auto;
    padding: 24px;
  }

  .language-card {
    min-width: 0;
  }
}
</style>
