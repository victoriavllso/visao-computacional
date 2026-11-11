"""The JSON export, in Portuguese like the labels in the photos: keys and
codes translated from the API's (which the page and the tests use).

    exportacao(entry) -> dict
"""

VEREDITOS = {
    "compliant": "de_acordo",
    "inconclusive": "inconclusivo",
    "non_compliant": "viola_as_regras",
    "info": "informativo",
}
MODOS = {"full": "frente_e_verso", "table_only": "so_tabela_nutricional"}
TIPOS_DE_FOTO = {
    "table": "tabela",
    "linear": "texto_corrido",
    "old_format": "formato_antigo",
    "no_table": "sem_tabela",
    "uncertain": "nao_identificado",
}
ALERTAS = {
    "added_sugars": "acucar_adicionado",
    "saturated_fat": "gordura_saturada",
    "sodium": "sodio",
}
TABELA = {
    "basis": "base",
    "added_sugars_g": "acucares_adicionados_g",
    "saturated_fat_g": "gorduras_saturadas_g",
    "sodium_mg": "sodio_mg",
}
VERSOES = {"back": "verso", "front": "frente", "rules": "regras"}


def _alertas(warnings: dict) -> dict:
    return {ALERTAS[key]: value for key, value in warnings.items()}


def exportacao(entry: dict) -> dict:
    result = entry["result"]
    return {
        "id": entry["id"],
        "criado_em": entry["created_at"],
        "nome": entry["name"],
        "resultado": {
            # Analyses saved before the table-only mode have no "mode".
            "modo": MODOS[result.get("mode", "full")],
            "veredito": VEREDITOS[result["verdict"]],
            "motivos": result["reasons"],
            "observacoes": result.get("notes", []),
            "tabela_nutricional": {
                TABELA[key]: value for key, value in result["nutrition"].items()
            },
            "alertas_exigidos": _alertas(result["required_warnings"]),
            "alertas_na_frente": _alertas(result["found_warnings"]),
            "foto_do_verso": TIPOS_DE_FOTO.get(
                result["back_photo_type"], result["back_photo_type"]
            ),
            "versoes": {
                VERSOES.get(key, key): value
                for key, value in result.get("versions", {}).items()
            },
            "simulado": result.get("mock", False),
        },
    }
