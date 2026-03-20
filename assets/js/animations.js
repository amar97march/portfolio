/**
 * Portfolio Animations — GSAP ScrollTrigger
 * Reusable animation functions for all pages
 */

(function () {
  "use strict";

  // Respect reduced motion
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    // Make all reveal elements visible immediately
    document.addEventListener("DOMContentLoaded", function () {
      document
        .querySelectorAll(
          ".reveal-up, .reveal-left, .reveal-right, .reveal-scale, .stagger-children > *, .split-text"
        )
        .forEach(function (el) {
          el.style.opacity = "1";
          el.style.transform = "none";
        });
    });
    return;
  }

  // Wait for GSAP to load
  function initAnimations() {
    if (typeof gsap === "undefined" || typeof ScrollTrigger === "undefined") {
      setTimeout(initAnimations, 100);
      return;
    }

    gsap.registerPlugin(ScrollTrigger);

    // Set default ease
    gsap.defaults({ ease: "power3.out", duration: 0.8 });

    // ===== REVEAL UP =====
    gsap.utils.toArray(".reveal-up").forEach(function (el) {
      gsap.fromTo(
        el,
        { y: 60, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.9,
          scrollTrigger: {
            trigger: el,
            start: "top 88%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // ===== REVEAL LEFT =====
    gsap.utils.toArray(".reveal-left").forEach(function (el) {
      gsap.fromTo(
        el,
        { x: -80, opacity: 0 },
        {
          x: 0,
          opacity: 1,
          duration: 0.9,
          scrollTrigger: {
            trigger: el,
            start: "top 88%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // ===== REVEAL RIGHT =====
    gsap.utils.toArray(".reveal-right").forEach(function (el) {
      gsap.fromTo(
        el,
        { x: 80, opacity: 0 },
        {
          x: 0,
          opacity: 1,
          duration: 0.9,
          scrollTrigger: {
            trigger: el,
            start: "top 88%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // ===== REVEAL SCALE =====
    gsap.utils.toArray(".reveal-scale").forEach(function (el) {
      gsap.fromTo(
        el,
        { scale: 0.85, opacity: 0 },
        {
          scale: 1,
          opacity: 1,
          duration: 0.8,
          scrollTrigger: {
            trigger: el,
            start: "top 88%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // ===== STAGGER CHILDREN =====
    gsap.utils.toArray(".stagger-children").forEach(function (container) {
      var children = container.children;
      gsap.fromTo(
        children,
        { y: 40, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.6,
          stagger: 0.1,
          scrollTrigger: {
            trigger: container,
            start: "top 85%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // ===== SPLIT TEXT (letter-by-letter) =====
    gsap.utils.toArray(".split-text").forEach(function (el) {
      var text = el.textContent;
      el.innerHTML = "";
      el.style.opacity = "1";

      text.split("").forEach(function (char) {
        var span = document.createElement("span");
        span.textContent = char === " " ? "\u00A0" : char;
        span.style.display = "inline-block";
        span.style.opacity = "0";
        el.appendChild(span);
      });

      gsap.to(el.querySelectorAll("span"), {
        opacity: 1,
        y: 0,
        duration: 0.05,
        stagger: 0.03,
        ease: "power2.out",
        scrollTrigger: {
          trigger: el,
          start: "top 85%",
          toggleActions: "play none none none",
        },
      });
    });

    // ===== PARALLAX =====
    gsap.utils.toArray(".parallax-bg").forEach(function (el) {
      var speed = parseFloat(el.dataset.speed) || 0.3;
      gsap.to(el, {
        yPercent: speed * 100,
        ease: "none",
        scrollTrigger: {
          trigger: el.parentElement || el,
          start: "top bottom",
          end: "bottom top",
          scrub: 1,
        },
      });
    });

    // ===== MAGNETIC BUTTONS =====
    document.querySelectorAll(".magnetic-btn").forEach(function (btn) {
      btn.addEventListener("mousemove", function (e) {
        var rect = btn.getBoundingClientRect();
        var x = e.clientX - rect.left - rect.width / 2;
        var y = e.clientY - rect.top - rect.height / 2;
        gsap.to(btn, {
          x: x * 0.3,
          y: y * 0.3,
          duration: 0.3,
          ease: "power2.out",
        });
      });
      btn.addEventListener("mouseleave", function () {
        gsap.to(btn, { x: 0, y: 0, duration: 0.5, ease: "elastic.out(1, 0.3)" });
      });
    });

    // ===== DRAW LINE (timeline connector) =====
    gsap.utils.toArray(".draw-line").forEach(function (line) {
      var length = line.getTotalLength ? line.getTotalLength() : 0;
      if (length > 0) {
        gsap.set(line, { strokeDasharray: length, strokeDashoffset: length });
        gsap.to(line, {
          strokeDashoffset: 0,
          ease: "none",
          scrollTrigger: {
            trigger: line.closest(".timeline") || line,
            start: "top 70%",
            end: "bottom 30%",
            scrub: 1,
          },
        });
      }
    });

    // ===== PROGRESS BARS =====
    gsap.utils.toArray(".animate-progress").forEach(function (bar) {
      var width = bar.dataset.progress || "0%";
      gsap.fromTo(
        bar,
        { width: "0%" },
        {
          width: width,
          duration: 1.2,
          ease: "power2.out",
          scrollTrigger: {
            trigger: bar,
            start: "top 90%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // ===== COUNTER ANIMATION =====
    gsap.utils.toArray(".counter").forEach(function (el) {
      var target = parseInt(el.dataset.target) || 0;
      gsap.to(el, {
        textContent: target,
        duration: 2,
        ease: "power1.out",
        snap: { textContent: 1 },
        scrollTrigger: {
          trigger: el,
          start: "top 90%",
          toggleActions: "play none none none",
        },
      });
    });

    // ===== VANILLA TILT INIT =====
    if (typeof VanillaTilt !== "undefined") {
      VanillaTilt.init(document.querySelectorAll(".tilt-card"), {
        max: 8,
        speed: 400,
        glare: true,
        "max-glare": 0.15,
        scale: 1.02,
      });
    }
  }

  // Initialize when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAnimations);
  } else {
    initAnimations();
  }
})();
