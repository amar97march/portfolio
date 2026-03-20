/**
 * Blog Table of Contents
 * - Finds all h2, h3 headings in .blog-content
 * - Generates TOC links in #toc-sidebar / #toc-nav / #toc-nav-mobile
 * - Uses IntersectionObserver to track active heading
 * - Mobile: TOC is collapsible via <details> element
 */
(function() {
  'use strict';

  function initTOC() {
    var content = document.getElementById('blog-content');
    var tocNav = document.getElementById('toc-nav');
    var tocNavMobile = document.getElementById('toc-nav-mobile');
    if (!content) return;

    var headings = content.querySelectorAll('h2, h3');
    if (headings.length === 0) return;

    // Build TOC HTML
    var tocHTML = '';
    headings.forEach(function(heading, i) {
      var id = heading.id || 'heading-' + i;
      heading.id = id;
      var level = heading.tagName.toLowerCase();
      var className = level === 'h3' ? 'toc-h3' : '';
      var text = heading.textContent.trim();
      tocHTML += '<a href="#' + id + '" class="' + className + '" data-target="' + id + '">' + text + '</a>';
    });

    if (tocNav) tocNav.innerHTML = tocHTML;
    if (tocNavMobile) tocNavMobile.innerHTML = tocHTML;

    // Collect all TOC links
    var tocLinks = [];
    if (tocNav) tocLinks = tocLinks.concat(Array.from(tocNav.querySelectorAll('a')));
    if (tocNavMobile) tocLinks = tocLinks.concat(Array.from(tocNavMobile.querySelectorAll('a')));

    if (tocLinks.length === 0) return;

    // Active section tracking with IntersectionObserver
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          var id = entry.target.id;
          tocLinks.forEach(function(link) {
            if (link.getAttribute('data-target') === id) {
              link.classList.add('active');
            } else {
              link.classList.remove('active');
            }
          });
        }
      });
    }, {
      rootMargin: '-80px 0px -60% 0px',
      threshold: 0
    });

    headings.forEach(function(heading) {
      observer.observe(heading);
    });

    // Smooth scroll for TOC links
    tocLinks.forEach(function(link) {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        var targetId = link.getAttribute('data-target');
        var target = document.getElementById(targetId);
        if (target) {
          var offset = 100;
          var top = target.getBoundingClientRect().top + window.scrollY - offset;
          window.scrollTo({ top: top, behavior: 'smooth' });

          // On mobile, collapse the details
          var details = link.closest('details');
          if (details) {
            setTimeout(function() { details.removeAttribute('open'); }, 300);
          }
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTOC);
  } else {
    initTOC();
  }
})();
