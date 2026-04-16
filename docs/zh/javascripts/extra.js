// Hetero-Paged-Infer Custom JavaScript

document.addEventListener('DOMContentLoaded', function() {
  // Add copy button to all code blocks without one
  document.querySelectorAll('pre > code').forEach(function(codeBlock) {
    if (!codeBlock.parentElement.querySelector('.md-clipboard')) {
      var button = document.createElement('button');
      button.className = 'md-clipboard';
      button.title = 'Copy to clipboard';
      button.innerHTML = '<span class="md-clipboard__icon"></span>';
      codeBlock.parentElement.appendChild(button);
    }
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
        // Update URL without jumping
        history.pushState(null, '', this.getAttribute('href'));
      }
    });
  });

  // Add language switcher if not present
  const headerSource = document.querySelector('.md-header__source');
  if (headerSource && !document.querySelector('.md-header__language')) {
    const langSwitcher = document.createElement('div');
    langSwitcher.className = 'md-header__language';
    
    const currentPath = window.location.pathname;
    const isEnglish = currentPath.includes('/en/') || !currentPath.includes('/zh/');
    
    langSwitcher.innerHTML = `
      <a href="/hetero-paged-infer/en/" class="${isEnglish ? 'active' : ''}">EN</a>
      <span style="color: rgba(255,255,255,0.5)">|</span>
      <a href="/hetero-paged-infer/zh/" class="${!isEnglish ? 'active' : ''}">中</a>
    `;
    
    headerSource.parentElement.insertBefore(langSwitcher, headerSource.nextSibling);
  }

  // Highlight active section in TOC
  const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -80% 0px',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        document.querySelectorAll('.md-nav__link--active').forEach(el => {
          el.classList.remove('md-nav__link--active');
        });
        const link = document.querySelector(`.md-nav__link[href="#${id}"]`);
        if (link) link.classList.add('md-nav__link--active');
      }
    });
  }, observerOptions);

  document.querySelectorAll('[id]').forEach(section => {
    observer.observe(section);
  });

  // Add performance chart animation
  const perfCards = document.querySelectorAll('.feature-card');
  const perfObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
      if (entry.isIntersecting) {
        setTimeout(() => {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }, index * 100);
      }
    });
  }, { threshold: 0.1 });

  perfCards.forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    perfObserver.observe(card);
  });

  // Terminal typing effect for hero code blocks
  const codeBlocks = document.querySelectorAll('.hero-code');
  codeBlocks.forEach(block => {
    const text = block.textContent;
    block.textContent = '';
    let i = 0;
    const typeInterval = setInterval(() => {
      if (i < text.length) {
        block.textContent += text.charAt(i);
        i++;
      } else {
        clearInterval(typeInterval);
      }
    }, 30);
  });
});

// Performance monitoring
if (window.performance) {
  window.addEventListener('load', function() {
    setTimeout(function() {
      const perfData = window.performance.timing;
      const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
      console.log('📊 Page load time:', pageLoadTime + 'ms');
    }, 0);
  });
}

// Service Worker registration for PWA (optional)
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    // Uncomment to enable offline support
    // navigator.serviceWorker.register('/hetero-paged-infer/sw.js');
  });
}
