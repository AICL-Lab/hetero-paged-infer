import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

// 动态 base path 处理
const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/')
      ? rawBase
      : `${rawBase}/`
    : `/${rawBase}/`
  : '/'

export default withMermaid(
  defineConfig({
    base,
    title: 'Hetero-Paged-Infer',
    description: 'High-performance LLM inference engine with PagedAttention and Continuous Batching',

    // 源码目录
    srcDir: '.',
    // 输出目录
    outDir: '.vitepress/dist',

    // Markdown 配置
    markdown: {
      lineNumbers: true,
      image: { lazyLoading: true }
    },

    // 多语言配置
    locales: {
      en: {
        label: 'English',
        lang: 'en-US',
        link: '/en/',
        themeConfig: {
          nav: [
            { text: 'Setup', link: '/en/setup/quickstart', activeMatch: '/en/setup/' },
            { text: 'Architecture', link: '/en/architecture/overview', activeMatch: '/en/architecture/' },
            { text: 'API', link: '/en/api/core-types', activeMatch: '/en/api/' },
            { text: 'Comparison', link: '/en/comparison/', activeMatch: '/en/comparison/' },
            { text: 'References', link: '/en/references/', activeMatch: '/en/references/' }
          ],
          sidebar: {
            '/en/setup/': [
              {
                text: 'Setup',
                items: [
                  { text: 'Quick Start', link: '/en/setup/quickstart' },
                  { text: 'Installation', link: '/en/setup/installation' },
                  { text: 'Configuration', link: '/en/setup/configuration' },
                  { text: 'Advanced Configuration', link: '/en/setup/advanced-config' }
                ]
              }
            ],
            '/en/architecture/': [
              {
                text: 'Architecture',
                items: [
                  { text: 'Overview', link: '/en/architecture/overview' },
                  { text: 'Design Principles', link: '/en/architecture/design' },
                  { text: 'PagedAttention', link: '/en/architecture/paged-attention' },
                  { text: 'Continuous Batching', link: '/en/architecture/continuous-batching' },
                  { text: 'Memory Management', link: '/en/architecture/memory-management' }
                ]
              }
            ],
            '/en/api/': [
              {
                text: 'API Reference',
                items: [
                  { text: 'Core Types', link: '/en/api/core-types' },
                  { text: 'Full Reference', link: '/en/api/reference' }
                ]
              }
            ],
            '/en/comparison/': [
              {
                text: 'Comparison',
                items: [
                  { text: 'Overview', link: '/en/comparison/' }
                ]
              }
            ],
            '/en/benchmarks/': [
              {
                text: 'Benchmarks',
                items: [
                  { text: 'Memory Efficiency', link: '/en/benchmarks/memory-efficiency' },
                  { text: 'Throughput', link: '/en/benchmarks/throughput' },
                  { text: 'Latency', link: '/en/benchmarks/latency' }
                ]
              }
            ],
            '/en/references/': [
              {
                text: 'References',
                items: [
                  { text: 'Papers', link: '/en/references/papers' },
                  { text: 'Projects', link: '/en/references/projects' }
                ]
              }
            ],
            '/en/deployment/': [
              {
                text: 'Deployment',
                items: [
                  { text: 'Docker Guide', link: '/en/deployment/docker' },
                  { text: 'Production Deploy', link: '/en/deployment/production' }
                ]
              }
            ],
            '/en/development/': [
              {
                text: 'Development',
                items: [
                  { text: 'Contributing', link: '/en/development/contributing' }
                ]
              }
            ]
          },
          editLink: {
            pattern: 'https://github.com/LessUp/hetero-paged-infer/edit/master/docs/:path',
            text: 'Edit this page on GitHub'
          },
          footer: {
            message: 'Released under the <a href="https://github.com/LessUp/hetero-paged-infer/blob/master/LICENSE">MIT License</a>.',
            copyright: '© 2024-present LessUp'
          },
          docFooter: {
            prev: 'Previous',
            next: 'Next'
          },
          outline: {
            label: 'On This Page',
            level: [2, 3]
          },
          lastUpdated: {
            text: 'Updated at'
          },
          search: {
            provider: 'local',
            options: {
              locales: {
                en: {
                  translations: {
                    button: {
                      buttonText: 'Search',
                      buttonAriaLabel: 'Search'
                    }
                  }
                }
              }
            }
          }
        }
      },
      zh: {
        label: '简体中文',
        lang: 'zh-CN',
        link: '/zh/',
        themeConfig: {
          nav: [
            { text: '安装配置', link: '/zh/setup/quickstart', activeMatch: '/zh/setup/' },
            { text: '架构设计', link: '/zh/architecture/overview', activeMatch: '/zh/architecture/' },
            { text: 'API 参考', link: '/zh/api/core-types', activeMatch: '/zh/api/' },
            { text: '项目对比', link: '/zh/comparison/', activeMatch: '/zh/comparison/' },
            { text: '参考文献', link: '/zh/references/', activeMatch: '/zh/references/' }
          ],
          sidebar: {
            '/zh/setup/': [
              {
                text: '安装配置',
                items: [
                  { text: '快速开始', link: '/zh/setup/quickstart' },
                  { text: '安装', link: '/zh/setup/installation' },
                  { text: '配置', link: '/zh/setup/configuration' },
                  { text: '高级配置', link: '/zh/setup/advanced-config' }
                ]
              }
            ],
            '/zh/architecture/': [
              {
                text: '架构设计',
                items: [
                  { text: '概览', link: '/zh/architecture/overview' },
                  { text: '设计原则', link: '/zh/architecture/design' },
                  { text: 'PagedAttention', link: '/zh/architecture/paged-attention' },
                  { text: '连续批处理', link: '/zh/architecture/continuous-batching' },
                  { text: '内存管理', link: '/zh/architecture/memory-management' }
                ]
              }
            ],
            '/zh/api/': [
              {
                text: 'API 参考',
                items: [
                  { text: '核心类型', link: '/zh/api/core-types' },
                  { text: '完整参考', link: '/zh/api/reference' }
                ]
              }
            ],
            '/zh/comparison/': [
              {
                text: '项目对比',
                items: [
                  { text: '概览', link: '/zh/comparison/' }
                ]
              }
            ],
            '/zh/benchmarks/': [
              {
                text: '性能基准',
                items: [
                  { text: '内存效率', link: '/zh/benchmarks/memory-efficiency' },
                  { text: '吞吐量', link: '/zh/benchmarks/throughput' },
                  { text: '延迟', link: '/zh/benchmarks/latency' }
                ]
              }
            ],
            '/zh/references/': [
              {
                text: '参考文献',
                items: [
                  { text: '论文引用', link: '/zh/references/papers' },
                  { text: '项目引用', link: '/zh/references/projects' }
                ]
              }
            ],
            '/zh/deployment/': [
              {
                text: '部署',
                items: [
                  { text: 'Docker 指南', link: '/zh/deployment/docker' },
                  { text: '生产部署', link: '/zh/deployment/production' }
                ]
              }
            ],
            '/zh/development/': [
              {
                text: '开发',
                items: [
                  { text: '贡献指南', link: '/zh/development/contributing' }
                ]
              }
            ]
          },
          editLink: {
            pattern: 'https://github.com/LessUp/hetero-paged-infer/edit/master/docs/:path',
            text: '在 GitHub 上编辑此页'
          },
          footer: {
            message: '基于 <a href="https://github.com/LessUp/hetero-paged-infer/blob/master/LICENSE">MIT 许可证</a> 发布。',
            copyright: '© 2024-present LessUp'
          },
          docFooter: {
            prev: '上一页',
            next: '下一页'
          },
          outline: {
            label: '本页目录',
            level: [2, 3]
          },
          lastUpdated: {
            text: '最后更新于'
          },
          search: {
            provider: 'local',
            options: {
              locales: {
                zh: {
                  translations: {
                    button: {
                      buttonText: '搜索',
                      buttonAriaLabel: '搜索'
                    }
                  }
                }
              }
            }
          }
        }
      }
    },

    // 全局主题配置
    themeConfig: {
      logo: '/images/logo.svg',
      siteTitle: 'Hetero-Paged-Infer',

      socialLinks: [
        { icon: 'github', link: 'https://github.com/LessUp/hetero-paged-infer' }
      ]
    },

    // Mermaid 配置
    mermaid: {
      // 参考: https://mermaid.js.org/config/theming.html
    },
    mermaidPlugin: {
      class: 'mermaid'
    },

    // 构建优化
    head: [
      ['link', { rel: 'icon', href: '/images/favicon.svg', type: 'image/svg+xml' }],
      ['meta', { name: 'theme-color', content: '#14b8a6' }],
      ['meta', { property: 'og:type', content: 'website' }],
      ['meta', { property: 'og:title', content: 'Hetero-Paged-Infer' }],
      ['meta', { property: 'og:description', content: 'High-performance LLM inference engine with PagedAttention and Continuous Batching' }],
      ['meta', { property: 'og:image', content: '/images/og-banner.svg' }]
    ],

    // Vite 配置
    vite: {
      plugins: [llmstxt()],
      logLevel: 'info'
    }
  })
)
