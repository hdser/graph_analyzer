WITH
signups AS (
    SELECT
        LOWER("user") AS avatar,
        "blockNumber" AS block_number,
        "timestamp",
        'Human' AS type,
        "token" AS own_token
    FROM "CrcV1_Signup"

    UNION ALL

    SELECT
        LOWER("organization") AS avatar,
        "blockNumber" AS block_number,
        "timestamp",
        'Org' AS type,
        NULL AS own_token
    FROM "CrcV1_OrganizationSignup"
),

avatar_info AS (
    SELECT
        avatar,
        MIN(block_number) AS block_number,
        MIN("timestamp") AS timestamp,
        (ARRAY_AGG(type ORDER BY "timestamp" ASC))[1] AS type,
        (array_remove(ARRAY_AGG(own_token), NULL))[1] AS own_token
    FROM signups
    GROUP BY avatar
),

avatar_names AS (
    SELECT
        LOWER(avatar) AS avatar,
        (array_remove(ARRAY_AGG("name" ORDER BY "timestamp" DESC), NULL))[1] AS name
    FROM "V_Crc_Avatars"
    WHERE version = 1
    GROUP BY LOWER(avatar)
),

tx AS (
    SELECT
        LOWER("from") AS account,
        LOWER("tokenAddress") AS "tokenAddress",
        - value AS delta
    FROM "CrcV1_Transfer"

    UNION ALL

    SELECT
        LOWER("to") AS account,
        LOWER("tokenAddress") AS "tokenAddress",
        value AS delta
    FROM "CrcV1_Transfer"
),

token_balances_raw AS (
    SELECT
        account AS avatar,
        "tokenAddress",
        SUM(delta) AS balance
    FROM tx
    WHERE account != '0x0000000000000000000000000000000000000000'
    GROUP BY account, "tokenAddress"
    HAVING SUM(delta) > 0
),

token_balances AS (
    SELECT
        avatar,
        ARRAY_AGG("tokenAddress") AS tokens,
        ARRAY_AGG(balance / POWER(10, 18)) AS tokens_balance,
        COUNT(*) AS tokens_cnt
    FROM token_balances_raw
    GROUP BY avatar
),

total_balances AS (
    SELECT
        avatar,
        SUM(balance) / POWER(10, 18) AS total_balance
    FROM token_balances_raw
    GROUP BY avatar
),

token_supply AS (
    SELECT
        LOWER("tokenAddress") AS avatar,
        SUM(delta) / POWER(10, 18) AS supply
    FROM tx
    WHERE account = '0x0000000000000000000000000000000000000000'
    GROUP BY LOWER("tokenAddress")
)

SELECT
    ai.avatar,
    ai.block_number,
    ai.timestamp,
    1 AS version,
    tb.tokens_cnt,
    tot.total_balance,
    ts.supply,
    an.name,
    ai.type,
    tb.tokens,
    tb.tokens_balance
FROM avatar_info ai
LEFT JOIN token_balances tb ON tb.avatar = ai.avatar
LEFT JOIN total_balances tot ON tot.avatar = ai.avatar
LEFT JOIN token_supply ts ON ts.avatar = ai.avatar
LEFT JOIN avatar_names an ON an.avatar = ai.avatar
