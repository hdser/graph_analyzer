SELECT
    LOWER(inviter) AS source,
    LOWER(avatar) AS target
FROM "CrcV2_RegisterHuman"
WHERE 
    "blockNumber" <= {block_number}
    AND inviter <> '0x0000000000000000000000000000000000000000'
    AND inviter <> avatar