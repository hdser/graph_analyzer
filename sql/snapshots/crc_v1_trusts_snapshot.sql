WITH latest_trust_events AS (
    SELECT
        "blockNumber",
        "timestamp",
        "transactionIndex",
        "logIndex",
        "user",
        "canSendTo",
        "limit"
    FROM (
        SELECT
            "CrcV1_Trust"."blockNumber",
            "CrcV1_Trust"."timestamp",
            "CrcV1_Trust"."transactionIndex",
            "CrcV1_Trust"."logIndex",
            "CrcV1_Trust"."user",
            "CrcV1_Trust"."canSendTo",
            "CrcV1_Trust"."limit",
            row_number() OVER (
                PARTITION BY "CrcV1_Trust"."user", "CrcV1_Trust"."canSendTo"
                ORDER BY "CrcV1_Trust"."blockNumber" DESC,
                         "CrcV1_Trust"."transactionIndex" DESC,
                         "CrcV1_Trust"."logIndex" DESC
            ) AS rn
        FROM "CrcV1_Trust"
        WHERE "CrcV1_Trust"."blockNumber" <= {block_number}
    ) t
    WHERE rn = 1
      AND "limit" > 0
),

human_avatars AS (
    SELECT "user" AS avatar
    FROM "CrcV1_Signup"
    WHERE "blockNumber" <= {block_number}
)

SELECT
    LOWER(t."user") AS source,
    LOWER(t."canSendTo") AS target
FROM latest_trust_events t
INNER JOIN human_avatars h1 ON h1.avatar = t."user"
INNER JOIN human_avatars h2 ON h2.avatar = t."canSendTo"
WHERE LOWER(t."user") != LOWER(t."canSendTo")
