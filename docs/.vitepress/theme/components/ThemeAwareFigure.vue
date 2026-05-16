<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useData, withBase } from 'vitepress'

const props = defineProps<{
  light: string
  dark: string
  alt: string
  caption?: string
}>()

const { isDark } = useData()
const hydrated = ref(false)

const resolveSrc = (src: string) => {
  if (/^(?:[a-z]+:|\/\/|#)/i.test(src)) {
    return src
  }

  return src.startsWith('/') ? withBase(src) : src
}

const currentSrc = computed(() =>
  resolveSrc(hydrated.value && isDark.value ? props.dark : props.light)
)

onMounted(() => {
  hydrated.value = true
})
</script>

<template>
  <figure class="theme-aware-figure">
    <img
      class="theme-aware-figure__image"
      :src="currentSrc"
      :alt="alt"
    />
    <figcaption v-if="caption">{{ caption }}</figcaption>
  </figure>
</template>
