from langgraph.graph import StateGraph, END
from typing import Literal, Union

# Düğümleri import et
from state import ResearchState
from nodes.manager import manager_node
from nodes.researcher import researcher_node
from nodes.filterer import filterer_node
from nodes.analyst import analyst_node
from nodes.reviewer import reviewer_node

# --- YÖNLENDİRME MANTIKLARI (ROUTERS) ---

def route_manager(state: ResearchState):
    """
    Manager konuyu netleştirdi mi?
    Evet -> Researcher'a git.
    Hayır -> END (Akışı bitir ki kullanıcıdan yeni girdi alabilelim)
    """
    protocol = state.get("research_protocol", {})
    
    if protocol.get("is_clear"):
        print("✅ Manager Onayı: Konu net, araştırmaya gidiliyor.")
        return "researcher"
    else:
        print("❓ Manager Sorusu: Konu net değil, kullanıcıya soruluyor.")
        # Buradan END döndürdüğümüzde, main.py'deki invoke işlemi biter 
        # ve son durumu döndürür. Orada mesajı kullanıcıya gösterebiliriz.
        return END

def route_filterer(state: ResearchState) -> Literal["manager", "analyst"]:
    """
    Filterer makaleleri beğenmezse Manager'a döner, beğenirse Analist'e gider.
    """
    feedback = state.get("filterer_feedback")
    revision = state.get("revision_number", 0)
    
    if feedback and revision < 3:
        print(f"🔄 Filterer reddetti ({revision+1}. deneme). Manager'a dönülüyor.")
        return "manager"
    else:
        return "analyst"

def route_analyst(state: ResearchState) -> Literal["analyst", "reviewer"]:
    """
    Analist raporu bitirdi mi?
    Evet -> Reviewer
    Hayır -> Analyst (Döngü)
    """
    if state.get("is_complete"):
        return "reviewer"
    else:
        return "analyst"

# Reviewer için routing — rapor kaldıysa ve deneme hakkı varsa analyst'e geri gönder
def route_reviewer(state: ResearchState) -> Union[str, object]:
    is_complete = state.get("is_complete", False)
    review_count = state.get("review_count", 0)
    
    if is_complete:
        print("✅ Reviewer Onayı: Rapor kabul edildi.")
        return END
    elif review_count < 3:
        print(f"🔄 Reviewer Reddi: Rapor analyst'e geri gönderiliyor (Deneme {review_count + 1}/3).")
        return "analyst"
    else:
        print("⚠️ Maksimum revizyon sayısına ulaşıldı. Rapor mevcut haliyle kabul ediliyor.")
        return END

# --- GRAFİK KURULUMU ---

def create_graph():
    workflow = StateGraph(ResearchState)

    # 1. Düğümleri Ekle
    workflow.add_node("manager", manager_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("filterer", filterer_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("reviewer", reviewer_node)

    # 2. Giriş Noktası
    workflow.set_entry_point("manager")

    # 3. Koşullu Kenarlar (Conditional Edges)
    
    # Manager Kararı (Sözlük kullanmıyoruz, otomatik eşleşme en güvenlisidir)
    workflow.add_conditional_edges(
        "manager",
        route_manager
    )
    
    # Filterer Kararı
    workflow.add_conditional_edges(
        "filterer",
        route_filterer
    )
    
    # Analyst Döngüsü
    workflow.add_conditional_edges(
        "analyst",
        route_analyst
    )
    
    # Reviewer Kararı (Onay veya revizyon)
    workflow.add_conditional_edges(
        "reviewer",
        route_reviewer
    )
    
    # 4. Normal Kenarlar
    workflow.add_edge("researcher", "filterer")

    # 5. Derle
    app = workflow.compile()
    return app