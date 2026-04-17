/* ==========================================================================
   Hetero-Paged-Infer — Unified Extra JavaScript
   Shared between English and Chinese MkDocs builds.
   ========================================================================== */

document$.subscribe(function () {
  // -- Copy button for code blocks ------------------------------------------
  document.querySelectorAll('.md-typeset pre').forEach(function (pre) {
    if (pre.querySelector('code') && !pre.querySelector('.copy-btn')) {
      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.textContent = 'Copy';
      btn.style.cssText =
        'position:absolute;top:4px;right:4px;padding:4px 8px;font-size:12px;' +
        'background:rgba(255,255,255,0.15);border:none;border-radius:4px;' +
        'color:var(--md-default-fg-color);cursor:pointer;opacity:0;transition:opacity 0.2s;';
      pre.style.position = 'relative';
      pre.appendChild(btn);
      pre.addEventListener('mouseenter', function () { btn.style.opacity = '1'; });
      pre.addEventListener('mouseleave', function () { btn.style.opacity = '0'; });
      btn.addEventListener('click', function () {
        var code = pre.querySelector('code').textContent;
        navigator.clipboard.writeText(code).then(function () {
          btn.textContent = 'Copied!';
          setTimeout(function () { btn.textContent = 'Copy'; }, 2000);
        });
      });
    }
  });

  // -- Smooth scrolling for anchor links ------------------------------------
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, null, this.getAttribute('href'));
      }
    });
  });

  // -- TOC active section highlighting via IntersectionObserver -------------
  var tocLinks = document.querySelectorAll('.md-nav__link');
  if ('IntersectionObserver' in window && tocLinks.length > 0) {
    var headings = document.querySelectorAll('h1[id], h2[id], h3[id]');
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            tocLinks.forEach(function (link) {
              link.classList.toggle(
                'md-nav__link--active',
                link.getAttribute('href') === '#' + entry.target.id
              );
            });
          }
        });
      },
      { rootMargin: '-20% 0px -80% 0px' }
    );
    headings.forEach(function (h) { observer.observe(h); });
  }

  // -- Feature card entrance animation --------------------------------------
  document.querySelectorAll('.feature-card').forEach(function (card, i) {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    setTimeout(function () {
      card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, 100 * i);
  });

  // -- Terminal typing effect for hero-code blocks --------------------------
  document.querySelectorAll('.hero-code').forEach(function (block) {
    var code = block.textContent;
    block.textContent = '';
    var i = 0;
    (function type() {
      if (i < code.length) {
        block.textContent += code.charAt(i);
        i++;
        setTimeout(type, 30);
      }
    })();
  });
});
