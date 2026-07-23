# UI Dependency Decision

Astro owns routing and static rendering. React islands are limited to controls
that require browser state. Fluent UI React v9 is used selectively for accessible
search inputs, command buttons, menus, drawers, tooltips, status messages, and
progress feedback.

Static paper rows, page layout, typography, charts, and topic treatments remain
Astro markup with project-owned CSS. This boundary avoids hydrating the archive,
preserves a distinct visual identity, and keeps Fluent UI where its interaction
and accessibility behavior is more valuable than custom controls.

Fluent UI runs in a `client:only="react"` island because its current package facade
does not prerender reliably under the Node 25/Astro 7 ESM path. A native HTML form
is supplied as the loading fallback, so search remains operable before hydration.
Lucide supplies icons because Fluent Icons 2.0.333 contains extensionless internal
ESM imports that fail during Astro's static build on Node 25.

Pagefind will provide the static search index in the next implementation stage.
