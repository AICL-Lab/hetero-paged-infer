import DefaultTheme from 'vitepress/theme'
import './style.css'
import FlagshipHero from './components/FlagshipHero.vue'
import ProofStrip from './components/ProofStrip.vue'
import SectionGrid from './components/SectionGrid.vue'
import ThemeAwareFigure from './components/ThemeAwareFigure.vue'
import ReferenceShelf from './components/ReferenceShelf.vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    DefaultTheme.enhanceApp?.({ app })
    app.component('FlagshipHero', FlagshipHero)
    app.component('ProofStrip', ProofStrip)
    app.component('SectionGrid', SectionGrid)
    app.component('ThemeAwareFigure', ThemeAwareFigure)
    app.component('ReferenceShelf', ReferenceShelf)
  },
}
