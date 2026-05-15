<script setup lang="ts">
import { VPLink } from 'vitepress/theme'

defineProps<{
  eyebrow: string
  title: string
  summary: string
  primaryText: string
  primaryHref: string
  secondaryText: string
  secondaryHref: string
}>()
</script>

<template>
  <section class="flagship-hero">
    <p class="flagship-eyebrow">{{ eyebrow }}</p>
    <h1 class="flagship-title">{{ title }}</h1>
    <p class="flagship-summary">{{ summary }}</p>
    <div class="flagship-actions">
      <VPLink class="flagship-action brand" :href="primaryHref" no-icon>{{ primaryText }}</VPLink>
      <VPLink class="flagship-action alt" :href="secondaryHref" no-icon>{{ secondaryText }}</VPLink>
    </div>
  </section>
</template>
