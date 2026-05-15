<script setup lang="ts">
import { VPLink } from 'vitepress/theme'

defineProps<{ cards: Array<{ title: string; summary: string; href: string }> }>()
</script>

<template>
  <div class="section-grid">
    <VPLink v-for="card in cards" :key="card.href" class="section-card" :href="card.href" no-icon>
      <h3>{{ card.title }}</h3>
      <p>{{ card.summary }}</p>
    </VPLink>
  </div>
</template>
