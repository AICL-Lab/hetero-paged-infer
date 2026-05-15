<script setup lang="ts">
import { VPLink } from 'vitepress/theme'

defineProps<{ entries: Array<{ title: string; kind: string; why: string; href: string }> }>()
</script>

<template>
  <div class="reference-shelf">
    <article v-for="entry in entries" :key="entry.href" class="reference-entry">
      <p class="reference-kind">{{ entry.kind }}</p>
      <h3><VPLink :href="entry.href" no-icon>{{ entry.title }}</VPLink></h3>
      <p>{{ entry.why }}</p>
    </article>
  </div>
</template>
