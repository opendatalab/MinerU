-- v002: Map legacy doclib tiers to the current public tier names.

UPDATE parses SET tier = 'medium' WHERE tier = 'standard';
UPDATE parses SET tier = 'high' WHERE tier = 'pro';

UPDATE fts_contents SET tier = 'medium' WHERE tier = 'standard';
UPDATE fts_contents SET tier = 'high' WHERE tier = 'pro';

UPDATE docs SET meta_tier = 'medium' WHERE meta_tier = 'standard';
UPDATE docs SET meta_tier = 'high' WHERE meta_tier = 'pro';

UPDATE parsing_rules SET tier = 'medium' WHERE tier = 'standard';
UPDATE parsing_rules SET tier = 'high' WHERE tier = 'pro';

UPDATE config SET value = 'medium' WHERE key = 'parse_server.local.managed_tier' AND value = 'standard';
UPDATE config SET value = 'high' WHERE key = 'parse_server.local.managed_tier' AND value = 'pro';
