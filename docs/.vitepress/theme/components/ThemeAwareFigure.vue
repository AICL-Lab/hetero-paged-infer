<script setup lang="ts">
import { withBase } from 'vitepress'

const props = defineProps<{
  light: string
  dark: string
  alt: string
  caption?: string
}>()

const resolveSrc = (src: string) => {
  if (/^(?:[a-z]+:|\/\/|#)/i.test(src)) {
    return src
  }

  return src.startsWith('/') ? withBase(src) : src
}
</script>

<template>
  <figure class="theme-aware-figure">
    <img class="theme-aware-figure__image theme-aware-figure__image--light" :src="resolveSrc(light)" :alt="alt" />
    <img class="theme-aware-figure__image theme-aware-figure__image--dark" :src="resolveSrc(dark)" :alt="alt" />
    <figcaption v-if="caption">{{ caption }}</figcaption>
  </figure>
</template>
