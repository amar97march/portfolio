/**
 * Portfolio Filter — Category-based project filtering
 * Uses GSAP for smooth show/hide animations when available,
 * falls back to CSS transitions otherwise.
 */
(function() {
  'use strict';

  function initFilter() {
    var filterBar = document.getElementById('portfolio-filter');
    var grid = document.getElementById('portfolio-grid');
    if (!filterBar || !grid) return;

    var buttons = filterBar.querySelectorAll('.filter-btn');
    var items = grid.querySelectorAll('[data-category]');

    buttons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        var filter = btn.dataset.filter;

        // Update active button
        buttons.forEach(function(b) { b.classList.remove('active'); });
        btn.classList.add('active');

        // Filter items
        items.forEach(function(item) {
          var category = item.dataset.category;
          var show = filter === 'all' || category === filter;

          if (show) {
            item.style.display = '';
            if (typeof gsap !== 'undefined') {
              gsap.fromTo(item,
                { opacity: 0, scale: 0.9, y: 20 },
                { opacity: 1, scale: 1, y: 0, duration: 0.4, ease: 'power2.out' }
              );
            } else {
              item.style.opacity = '1';
              item.style.transform = 'none';
            }
          } else {
            if (typeof gsap !== 'undefined') {
              gsap.to(item, {
                opacity: 0,
                scale: 0.9,
                duration: 0.3,
                ease: 'power2.in',
                onComplete: function() {
                  item.style.display = 'none';
                }
              });
            } else {
              item.style.display = 'none';
            }
          }
        });

        // Update URL hash
        if (history.replaceState) {
          history.replaceState(null, null, filter === 'all' ? location.pathname : '#' + filter);
        }
      });
    });

    // Check for hash on load
    var hash = location.hash.replace('#', '');
    if (hash) {
      var matchBtn = filterBar.querySelector('[data-filter="' + hash + '"]');
      if (matchBtn) matchBtn.click();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFilter);
  } else {
    initFilter();
  }
})();
