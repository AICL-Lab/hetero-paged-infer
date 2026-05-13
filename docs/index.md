---
layout: home
---

<script setup>
import { onMounted } from 'vue'

// 自动检测浏览器语言并跳转
onMounted(() => {
  const savedLang = localStorage.getItem('prefer-language')
  if (savedLang) {
    window.location.href = savedLang === 'zh' ? '/zh/' : '/en/'
    return
  }

  const browserLang = navigator.language.toLowerCase()
  if (browserLang.startsWith('zh')) {
    window.location.href = '/zh/'
  } else {
    window.location.href = '/en/'
  }
})
</script>

<div class="language-select">
  <div class="language-card" onclick="window.location.href='/en/'">
    <div class="language-icon">🇺🇸</div>
    <div class="language-name">English</div>
    <div class="language-desc">View documentation in English</div>
  </div>
  <div class="language-card" onclick="window.location.href='/zh/'">
    <div class="language-icon">🇨🇳</div>
    <div class="language-name">简体中文</div>
    <div class="language-desc">查看中文文档</div>
  </div>
</div>

<style>
.language-select {
  display: flex;
  gap: 24px;
  justify-content: center;
  align-items: center;
  min-height: 60vh;
  padding: 48px;
}

.language-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 32px 48px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.language-card:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand-1);
  box-shadow: var(--shadow-lg);
}

.language-icon {
  font-size: 3rem;
}

.language-name {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.language-desc {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
}

@media (max-width: 639px) {
  .language-select {
    flex-direction: column;
    min-height: auto;
    padding: 24px;
  }
}
</style>
