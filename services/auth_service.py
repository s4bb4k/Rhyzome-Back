from supabase_client import supabase

def login_user(email, password):

    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    return response


def register_user(email, password):

    response = supabase.auth.sign_up({
        "email": email,
        "password": password
    })

    return response