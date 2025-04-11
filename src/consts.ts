// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

// Base Page Metadata, src/layouts/BaseLayout.astro
export const BRAND_NAME = "AuroraLM";
export const SITE_TITLE = "AuroraLM";
export const SITE_DESCRIPTION = "Fully Permissive Open Multiodel LLMs";

// Tags Page Metadata, src/pages/tags/index.astro
export const Tags_TITLE = "AuroraLM - All Tags";
export const Tags_DESCRIPTION =
  "AuroraLM - All tags and the count of articles related to each tag";

// Tags Page Metadata, src/pages/tags/[tag]/[page].astro
export function getTagMetadata(tag: string) {
  return {
    title: `All articles on '${tag}' tag in AuroraLM`,
    description: `Explore articles about ${tag} for different perspectives and in-depth analysis.`,
  };
}

// Category Page Metadata, src/pages/category/[category]/[page].astro
export function getCategoryMetadata(category: string) {
  return {
    title: `All articles in '${category}' category in AuroraLM`,
    description: `Browse all articles under the ${category} category in AuroraLM`,
  };
}

// Header Links, src/components/Header.astro
export const HeaderLinks = [
  { href: "/posts/about-us/", title: "About Us" },
  { href: "/category/One/1/", title: "Blog Posts" },
  { href: "/tags/", title: "Sort by Tags" },
];

// Footer Links, src/components/Footer.astro
export const FooterLinks = [
  { href: "https://www.ontocord.ai/", title: "OntocordAI" },
];

// Social Links, src/components/Footer.astro
export const SocialLinks = [
  {
    href: "https://x.com/ontocord",
    icon: "tabler:brand-twitter",
    label: "Twitter",
  },
  {
    href: "https://github.com/aurora-lm",
    icon: "tabler:brand-github",
    label: "GitHub",
  },
  {
    href: "https://discord.gg/RBAjeWSA", // Replace with your actual Discord invite link
    icon: "tabler:brand-discord",
    label: "Discord",
  },
];

// Search Page Metadata, src/pages/search.astro
export const SEARCH_PAGE_TITLE = `${SITE_TITLE} - Site Search`;
export const SEARCH_PAGE_DESCRIPTION = `Search all content on ${SITE_TITLE}`;
