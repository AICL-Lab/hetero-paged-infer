/**
 * 同步 CHANGELOG.md 到文档站点
 *
 * 将根目录的 CHANGELOG.md 同步到 docs/en/changelog/index.md 和 docs/zh/changelog/index.md
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const rootDir = resolve(__dirname, '..', '..')
const changelogPath = resolve(rootDir, 'CHANGELOG.md')

// 读取 CHANGELOG.md
if (!existsSync(changelogPath)) {
  console.log('CHANGELOG.md not found, skipping sync')
  process.exit(0)
}

const changelog = readFileSync(changelogPath, 'utf-8')

// 转换格式
// ## [0.1.0] - 2024-01-15 → ## 0.1.0 (2024-01-15)
let formatted = changelog.replace(
  /## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})/g,
  '## $1 ($2)'
)

// 移除 ### Added/Changed/Fixed 等子标题
formatted = formatted.replace(/### (Added|Changed|Fixed|Deprecated|Removed|Security)\n/g, '')

// 添加 VitePress frontmatter
const englishContent = `---
title: Changelog
---

# Changelog

${formatted}
`

const chineseContent = `---
title: 变更日志
---

# 变更日志

${formatted}
`

// 确保目录存在
const enChangelogDir = resolve(rootDir, 'docs', 'en', 'changelog')
const zhChangelogDir = resolve(rootDir, 'docs', 'zh', 'changelog')

if (!existsSync(enChangelogDir)) {
  mkdirSync(enChangelogDir, { recursive: true })
}
if (!existsSync(zhChangelogDir)) {
  mkdirSync(zhChangelogDir, { recursive: true })
}

// 写入文件
writeFileSync(resolve(enChangelogDir, 'index.md'), englishContent)
writeFileSync(resolve(zhChangelogDir, 'index.md'), chineseContent)

console.log('Changelog synced successfully!')
