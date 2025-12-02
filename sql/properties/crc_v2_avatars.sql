WITH 
tx AS (
	SELECT "CrcV2_TransferSingle"."timestamp",
		"CrcV2_TransferSingle"."from" AS account,
		"CrcV2_TransferSingle"."tokenAddress",
		"CrcV2_TransferSingle".id,
		- "CrcV2_TransferSingle".value AS delta
	   FROM "CrcV2_TransferSingle"
	UNION ALL
	 SELECT "CrcV2_TransferSingle"."timestamp",
		"CrcV2_TransferSingle"."to" AS account,
		"CrcV2_TransferSingle"."tokenAddress",
		"CrcV2_TransferSingle".id,
		"CrcV2_TransferSingle".value AS delta
	   FROM "CrcV2_TransferSingle"
	UNION ALL
	 SELECT "CrcV2_TransferBatch"."timestamp",
		"CrcV2_TransferBatch"."from" AS account,
		"CrcV2_TransferBatch"."tokenAddress",
		"CrcV2_TransferBatch".id,
		- "CrcV2_TransferBatch".value AS delta
	   FROM "CrcV2_TransferBatch"
	UNION ALL
	 SELECT "CrcV2_TransferBatch"."timestamp",
		"CrcV2_TransferBatch"."to" AS account,
		"CrcV2_TransferBatch"."tokenAddress",
		"CrcV2_TransferBatch".id,
		"CrcV2_TransferBatch".value AS delta
	   FROM "CrcV2_TransferBatch"
), 
agg AS (
	SELECT 
		tx.account,
		tx.id,
		tx."tokenAddress",
		sum(tx.delta) AS balance,
		max(tx."timestamp") AS last_ts
	FROM tx
	GROUP BY tx.account, tx.id, tx."tokenAddress"
),

token_supply AS (
	SELECT 
		"tokenAddress" AS avatar
		,SUM(floor(crc_demurrage(1675209600::bigint, last_ts, -balance)))/POWER(10,18) AS supply
	FROM agg
	WHERE account = '0x0000000000000000000000000000000000000000'::text
	GROUP BY 1
),

token_balances AS (
	SELECT 
	 	account AS avatar
	    ,ARRAY_AGG("tokenAddress") AS tokens
	    ,ARRAY_AGG(
			floor(crc_demurrage(1675209600::bigint, last_ts,balance))/POWER(10,18)
		) AS tokens_balance
		,COUNT(*) AS tokens_cnt
	FROM agg 
	WHERE account <> '0x0000000000000000000000000000000000000000'::text AND balance > POWER(10,18)
	GROUP BY 1
),

total_balances AS (
	SELECT 
	 	account AS avatar
	    ,SUM(floor(crc_demurrage(1675209600::bigint, last_ts, balance)))/POWER(10,18) AS total_balance
	FROM agg
	WHERE account <> '0x0000000000000000000000000000000000000000'::text AND balance > 0::numeric
	GROUP BY 1
),

ipfs_data AS (
	SELECT * FROM "ipfs_files"
),

last_metadata AS (
	SELECT 
		avatar
		,(ARRAY_AGG("metadataDigest" ORDER BY timestamp DESC))[1] AS "metadataDigest"
	FROM "CrcV2_UpdateMetadataDigest" 
	GROUP BY 1
),

avatars AS (
	SELECT 
		avatar
		,MIN("blockNumber") AS "blockNumber"
		,MIN(timestamp) AS timestamp
		,MIN(version) AS version
		,(ARRAY_AGG(type ORDER BY timestamp DESC))[1] AS "type"
		,(array_remove(ARRAY_AGG("name" ORDER BY timestamp DESC), NULL))[1] AS "name"
		,(array_remove(ARRAY_AGG("cidV0Digest" ORDER BY timestamp DESC), NULL))[1] AS "cidV0Digest"
	FROM "V_Crc_Avatars"
	GROUP BY 1
),

avatars_metadata AS (
	SELECT 
		t1."blockNumber" AS block_number
		,t1.timestamp
		,t1.avatar
		,t1.version
		,CASE
			WHEN t1.type = 'CrcV2_RegisterHuman' THEN 'Human'
			WHEN t1.type = 'CrcV2_RegisterOrganization' THEN 'Org'
			ELSE 'Group'
		END AS type
		,COALESCE(t1.name, trim(both '"' from (t2.payload->'name')::text)) AS name
	FROM avatars t1
	LEFT JOIN  ipfs_data  t2
		ON t2.metadata_digest = t1."cidV0Digest"
	LEFT JOIN
		last_metadata t3
		ON t3.avatar = t1.avatar
		AND t3."metadataDigest" = t2.metadata_digest
    INNER JOIN "V_CrcV2_Avatars" t4
		ON t4.avatar = t1.avatar
)

SELECT 
 	t1.avatar
	,t4.block_number
	,t4.timestamp
	,t4.version
	,t1.tokens_cnt
	,t3.total_balance
	,t2.supply
	,t4.name
	,t4.type
	,t1.tokens
    ,t1.tokens_balance
FROM token_balances t1
LEFT JOIN token_supply t2
   	ON t2.avatar = t1.avatar
LEFT JOIN total_balances t3
	ON t3.avatar = t1.avatar
INNER JOIN avatars_metadata t4
	ON t4.avatar = t1.avatar