<script setup lang="ts">
defineProps<{ items: Array<{ label: string; value: string; detail: string }> }>()
</script>

<template>
  <div class="proof-strip">
    <article v-for="item in items" :key="item.label" class="proof-card">
      <div class="proof-value">{{ item.value }}</div>
      <div class="proof-label">{{ item.label }}</div>
      <p class="proof-detail">{{ item.detail }}</p>
    </article>
  </div>
</template>
