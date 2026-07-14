from playwright.sync_api import sync_playwright

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    page = browser.new_page()

    page.goto(
        "file:///Users/grapefruit/Desktop/Guru-Model2.0/bigjobby.html"
    )

    # wait for MediaPipe/model loading
    page.wait_for_timeout(5000)


    # check functions exist
    result = page.evaluate("""
    () => ({
        processImage: typeof processImage,
        bbox: typeof getRegionBBox,
        blit: typeof blitFlippedRegion
    })
    """)

    print(result)

    browser.close()