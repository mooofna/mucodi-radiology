import { resolve } from '$app/paths';

/** Site root including the deployment base path, without a trailing slash. */
export function siteRoot(): string {
  return resolve('/').replace(/\/$/, '');
}
