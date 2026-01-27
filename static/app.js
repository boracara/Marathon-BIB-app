;(() => {
  const grid = document.getElementById("grid")
  const toggleNamesBtn = document.getElementById("toggleNames")
  const openAllBtn = document.getElementById("openAll")

  // Lightbox
  const lightbox = document.getElementById("lightbox")
  const lbClose = document.getElementById("lbClose")
  const lbX = document.getElementById("lbX")
  const lbImg = document.getElementById("lbImg")
  const lbTitle = document.getElementById("lbTitle")
  const lbCount = document.getElementById("lbCount")
  const lbOpen = document.getElementById("lbOpen")
  const lbDownload = document.getElementById("lbDownload")
  const lbPrev = document.getElementById("lbPrev")
  const lbNext = document.getElementById("lbNext")

  // Build gallery from tiles (original photos only)
  let gallery = []
  let currentIndex = -1

  function buildGallery() {
    if (!grid) return

    const tiles = grid.querySelectorAll(".tile")
    gallery = Array.from(tiles)
      .map((tile) => {
        const img = tile.querySelector(".tile__img")
        const filename = tile.getAttribute("data-filename") || ""
        const downloadHref = tile.getAttribute("data-download") || "#"
        return {
          src: img ? img.src : "",
          filename,
          downloadHref,
        }
      })
      .filter((x) => x.src)
  }

  function renderLightbox() {
    if (currentIndex < 0 || currentIndex >= gallery.length) return

    const item = gallery[currentIndex]
    lbImg.src = item.src
    lbTitle.textContent = ` `;

    lbOpen.href = item.src

    // Use your actual download route
    lbDownload.href = item.downloadHref

    // Counter
    if (lbCount) {
      lbCount.textContent = gallery.length > 1 ? `Photo ${currentIndex + 1} of ${gallery.length}` : ""
    }

    // Prev/Next enable state
    const showNav = gallery.length > 1
    if (lbPrev) lbPrev.style.display = showNav ? "inline-flex" : "none"
    if (lbNext) lbNext.style.display = showNav ? "inline-flex" : "none"

    if (lbPrev) {
      lbPrev.style.display = showNav ? "flex" : "none"
      lbPrev.style.opacity = showNav ? "1" : "0"
    }
    if (lbNext) {
      lbNext.style.display = showNav ? "flex" : "none"
      lbNext.style.opacity = showNav ? "1" : "0"
    }
  }

  function openLightboxAt(index) {
    if (!gallery.length) buildGallery()
    if (index < 0 || index >= gallery.length) return

    currentIndex = index
    lightbox.classList.add("isOpen")
    lightbox.setAttribute("aria-hidden", "false")
    renderLightbox()
  }

  function closeLightbox() {
    lightbox.classList.remove("isOpen")
    lightbox.setAttribute("aria-hidden", "true")
    lbImg.src = ""
    lbOpen.href = "#"
    lbDownload.href = "#"
    currentIndex = -1
  }

  function nextImage() {
    if (gallery.length <= 1) return
    currentIndex = (currentIndex + 1) % gallery.length
    renderLightbox()
  }

  function prevImage() {
    if (gallery.length <= 1) return
    currentIndex = (currentIndex - 1 + gallery.length) % gallery.length
    renderLightbox()
  }

  // Click on grid -> open lightbox at correct index
  if (grid) {
    buildGallery()

    grid.addEventListener("click", (e) => {
      const btn = e.target.closest(".tile__imgBtn")
      if (!btn) return

      const tile = btn.closest(".tile")
      const tiles = Array.from(grid.querySelectorAll(".tile"))
      const index = tiles.indexOf(tile)
      if (index >= 0) openLightboxAt(index)
    })
  }

  // Toggle filenames
  if (toggleNamesBtn) {
    toggleNamesBtn.addEventListener("click", () => {
      document.body.classList.toggle("showNames")
    })
  }

  // Open all images in new tabs
  if (openAllBtn && grid) {
    openAllBtn.addEventListener("click", () => {
      const imgs = grid.querySelectorAll(".tile img")
      imgs.forEach((img) => window.open(img.src, "_blank"))
    })
  }

  // Lightbox controls
  if (lbClose) lbClose.addEventListener("click", closeLightbox)
  if (lbX) lbX.addEventListener("click", closeLightbox)
  if (lbPrev) lbPrev.addEventListener("click", prevImage)
  if (lbNext) lbNext.addEventListener("click", nextImage)

  // Keyboard: Esc + arrows
  document.addEventListener("keydown", (e) => {
    if (!lightbox.classList.contains("isOpen")) return

    if (e.key === "Escape") closeLightbox()
    if (e.key === "ArrowRight") nextImage()
    if (e.key === "ArrowLeft") prevImage()
  })

  // Swipe support (touch)
  let touchStartX = 0
  let touchStartY = 0
  let touchActive = false

  function onTouchStart(ev) {
    if (!lightbox.classList.contains("isOpen")) return
    const t = ev.touches && ev.touches[0]
    if (!t) return

    touchActive = true
    touchStartX = t.clientX
    touchStartY = t.clientY
  }

  function onTouchEnd(ev) {
    if (!touchActive || !lightbox.classList.contains("isOpen")) return
    touchActive = false

    const t = ev.changedTouches && ev.changedTouches[0]
    if (!t) return

    const dx = t.clientX - touchStartX
    const dy = t.clientY - touchStartY

    // Horizontal swipe threshold
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy)) {
      if (dx < 0) nextImage()
      else prevImage()
    }
  }

  if (lbImg) {
    lbImg.addEventListener("touchstart", onTouchStart, { passive: true })
    lbImg.addEventListener("touchend", onTouchEnd, { passive: true })
  }
})()
