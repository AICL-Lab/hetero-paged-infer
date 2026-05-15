<script setup lang="ts">
import { withBase } from 'vitepress'

defineProps<{ entries: Array<{ title: string; kind: string; why: string; href: string }> }>()

const resolveHref = (href: string) => {
  if (/^(?:[a-z]+:|\/\/|#)/i.test(href)) {
    return href
  }

  return href.startsWith('/') ? withBase(href) : href
}
</script>

<template>
  <div class="reference-shelf">
    <article v-for="entry in entries" :key="entry.href" class="reference-entry">
      <p class="reference-kind">{{ entry.kind }}</p>
      <h3><a :href="resolveHref(entry.href)">{{ entry.title }}</a></h3>
      <p>{{ entry.why }}</p>
    </article>
  </div>
</template>
