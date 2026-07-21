-- v002: Make databases containing retired Tier values safe for the renamed Tier system.
-- Parsed artifacts are intentionally not migrated, so rows referencing legacy cache directories are removed.

DELETE FROM parses
WHERE tier NOT IN ('flash', 'basic', 'standard', 'advanced');

DELETE FROM fts_contents
WHERE tier IS NOT NULL
  AND tier NOT IN ('flash', 'basic', 'standard', 'advanced');

UPDATE docs
SET meta_tier = NULL
WHERE meta_tier IS NOT NULL
  AND meta_tier NOT IN ('flash', 'basic', 'standard', 'advanced');

UPDATE parsing_rules
SET tier = CASE tier
    WHEN 'medium' THEN 'basic'
    WHEN 'high' THEN 'standard'
    WHEN 'xhigh' THEN 'advanced'
    ELSE NULL
END
WHERE tier IS NOT NULL
  AND tier NOT IN ('flash', 'basic', 'standard', 'advanced');

UPDATE config
SET value = CASE value
    WHEN 'medium' THEN 'basic'
    WHEN 'high' THEN 'standard'
    WHEN 'xhigh' THEN 'standard'
END
WHERE key = 'parse_server.local.managed_tier'
  AND value IN ('medium', 'high', 'xhigh');

DELETE FROM config
WHERE key = 'parse_server.local.managed_tier'
  AND value NOT IN ('basic', 'standard');
