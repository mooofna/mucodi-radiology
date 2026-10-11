<script lang="ts">
  // The evidence page was renamed; links from before keep working.
  import { resolve } from '$app/paths';
</script>

<svelte:head>
  <title>Results, MuCoDi Chest CT</title>
  <meta http-equiv="refresh" content={`0; url=${resolve('/results')}`} />
  <link rel="canonical" href={resolve('/results')} />
</svelte:head>

<div class="page">
  <h1>Results</h1>
  <p>This page has moved to <a href={resolve('/results')}>Results</a>.</p>
</div>
