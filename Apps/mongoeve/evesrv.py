from eve import Eve
from eve.auth import TokenAuth
from flask import current_app as app
from waitress import serve

class TokenAuth(TokenAuth):
    def check_auth(self, token, allowed_roles, resource, method):
        """For the purpose of this example the implementation is as simple as
        possible. A 'real' token should probably contain a hash of the
        username/password combo, which sould then validated against the account
        data stored on the DB.
        """
        # use Eve's own db driver; no additional connections/resources are used
        accounts = app.data.driver.db['accounts']
        return accounts.find_one({'token': token})


if __name__ == '__main__':
    app = Eve(auth=TokenAuth)
    serve(app,host="0.0.0.0",port=2308)
