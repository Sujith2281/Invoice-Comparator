from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_invoices, name='upload_invoices'),
    path('match-product/', views.match_product_view, name='compare_product'),
    path('diff-identifier/', views.diff_identifier, name='diff_identifier'),
    path('row-comparator/',views.row_comparator,name="row-comaparator"),
    path('get-product-keys/',views.get_product_keys,name = "item keys"),
    path('column-comparator/',views.column_comparator,name='column-comparator'),
    path('get-dropdown-columns/',views.get_dropdown_columns,name='get-dropdown-columns'),
    path('compare-selected-columns/',views.compare_selected_columns,name='compare-selected-columns'),
    path('build-mixed-table/',views.build_mixed_table,name='build-mixed-table'),
    path('create-unified-table/',views.create_unified_table, name='create-unified-table'),
    path('final-result-mapper/',views.final_result_mapper, name='final-result-mapper')
]

#path("compare/", views.compare_invoices, name="compare_invoices"),
#path("match/", views.fuzzy_match_items, name="fuzzy_match_items"),  # Example view