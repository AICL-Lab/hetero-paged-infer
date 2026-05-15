<script setup lang="ts">
import { withBase } from 'vitepress'

defineProps<{
  eyebrow: string
  title: string
  summary: string
  primaryText: string
  primaryHref: string
  secondaryText: string
  secondaryHref: string
}>()

const resolveHref = (href: string) => {
  if (/^(?:[a-z]+:|\/\/|#)/i.test(href)) {
    return href
  }

  return href.startsWith('/') ? withBase(href) : href
}
</script>

<template>
  <section class="flagship-hero">
    <p class="flagship-eyebrow">{{ eyebrow }}</p>
    <h1 class="flagship-title">{{ title }}</h1>
    <p class="flagship-summary">{{ summary }}</p>
    <div class="flagship-actions">
      <a class="flagship-action brand" :href="resolveHref(primaryHref)">{{ primaryText }}</a>
      <a class="flagship-action alt" :href="resolveHref(secondaryHref)">{{ secondaryText }}</a>
    </div>
  </section>
</template>
