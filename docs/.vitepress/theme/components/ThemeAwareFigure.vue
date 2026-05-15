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
    <img
      class="theme-aware-figure__image"
      data-theme-aware-image
      :src="resolveSrc(light)"
      :data-theme-src-light="resolveSrc(light)"
      :data-theme-src-dark="resolveSrc(dark)"
      :alt="alt"
    />
    <figcaption v-if="caption">{{ caption }}</figcaption>
  </figure>
</template>
