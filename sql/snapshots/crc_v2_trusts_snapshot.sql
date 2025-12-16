WITH v_crcv2_trustrelations AS (
    SELECT 
        "blockNumber",
        "timestamp",
        "transactionIndex",
        "logIndex",
        "transactionHash",
        trustee,
        truster,
        "expiryTime"
    FROM ( 
        SELECT 
            "CrcV2_Trust"."blockNumber",
            "CrcV2_Trust"."timestamp",
            "CrcV2_Trust"."transactionIndex",
            "CrcV2_Trust"."logIndex",
            "CrcV2_Trust"."transactionHash",
            "CrcV2_Trust".truster,
            "CrcV2_Trust".trustee,
            "CrcV2_Trust"."expiryTime",
            row_number() OVER (
                PARTITION BY "CrcV2_Trust".truster, "CrcV2_Trust".trustee 
                ORDER BY "CrcV2_Trust"."blockNumber" DESC, 
                         "CrcV2_Trust"."transactionIndex" DESC, 
                         "CrcV2_Trust"."logIndex" DESC
            ) AS rn
        FROM "CrcV2_Trust"
        WHERE "CrcV2_Trust"."blockNumber" <= {block_number}
    ) t
    WHERE 
        rn = 1 
        AND "expiryTime" > (
            SELECT "timestamp" 
            FROM "System_Block" 
            WHERE "blockNumber" = {block_number}
        )::numeric
)

SELECT 
    LOWER(trustee) AS source,
    LOWER(truster) AS target
FROM v_crcv2_trustrelations
WHERE truster != trustee