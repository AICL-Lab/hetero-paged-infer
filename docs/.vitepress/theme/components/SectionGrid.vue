<script setup lang="ts">
import { withBase } from 'vitepress'

defineProps<{ cards: Array<{ title: string; summary: string; href: string }> }>()

const resolveHref = (href: string) => {
  if (/^(?:[a-z]+:|\/\/|#)/i.test(href)) {
    return href
  }

  return href.startsWith('/') ? withBase(href) : href
}
</script>

<template>
  <div class="section-grid">
    <a v-for="card in cards" :key="card.href" class="section-card" :href="resolveHref(card.href)">
      <h3>{{ card.title }}</h3>
      <p>{{ card.summary }}</p>
    </a>
  </div>
</template>
